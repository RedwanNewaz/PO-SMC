-- GLOBAL VARIABLE 
manipulator={
    ArmJoints={
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'upperarm_roll_joint',
        'elbow_flex_joint',
        'forearm_roll_joint',
        'wrist_flex_joint',
        'wrist_roll_joint'
    },
    jh={-1,-1,-1,-1,-1,-1,-1},
    ikGroup=sim.getIkGroupHandle('fetch'),
    ikTarget=sim.getObjectHandle('StartLocation'),
    collisionPairs={sim.getCollectionHandle('Manipulator'),sim.getCollectionHandle('Env'),sim.getCollectionHandle('Body')},
    --collisionPairs={sim.getObjectHandle('Env')},
    
    target1=sim.getObjectHandle('target#0'),
    target2=sim.getObjectHandle('target#1'),
    
    approachDirectionObstacle=sim.getObjectHandle('approachDirectionObstacle'),
     metric={0.5,1,1,0.5,0.1,0.2,0.1},
    --metric={0.2,1,0.8,0.1,0.1,0.1},
    forbidLevel=0,
    ikShift=-0.1,
    ikSteps=50,
    
}

function manipulator:new(o)    
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    for i=1,7,1 do
        self.jh[i]=sim.getObjectHandle(self.ArmJoints[i])
    end
    -- self.allTargets={self.target1,self.target2,self.target3,self.target4}
    self.allTargets={self.target2,self.target1}
    self.targetIndex=1
    return o
end

function manipulator:defaultPose()
    vel=110
    accel=40
    jerk=80
    local currentVel={0,0,0,0,0,0,0}
    local currentAccel={0,0,0,0,0,0,0}
    local targetVel = {0,0,0,0,0,0,0}
    local jointHandles = {}
    local maxVel ={}
    local maxAccel ={}
    local maxJerk = {}
    for i=1,7,1 do
        maxVel[i] = vel*math.pi/180
        maxAccel[i] = accel*math.pi/180
        maxJerk[i] = jerk*math.pi/180
    end
    print(jointHandles)
    targetPos1={90*math.pi/180,90*math.pi/180,170*math.pi/180,-90*math.pi/180,90*math.pi/90,90*math.pi/180,0}
    sim.rmlMoveToJointPositions(self.jh,-1,currentVel,currentAccel,maxVel,maxAccel,maxJerk,targetPos1,targetVel)

end

function manipulator:run()

    -- This is the main loop. We move from one target to the next
    local theTarget=self.allTargets[self.targetIndex]
    self.targetIndex=self.targetIndex+1
    if self.targetIndex>#self.allTargets then
        self.targetIndex=1
    end

    -- m is the transformation matrix or pose of the current target:
    local m=sim.getObjectMatrix(theTarget,-1)

    -- Compute a pose that is shifted by ikDist along the Z-axis of pose m,
    -- so that we have a final approach that is linear along target axis Z:
    m=self:getMatrixShiftedAlongZ(m,-self.ikShift)

    -- Find several configs for pose m, and order them according to the
    -- distance to current configuration (smaller distance is better).
    -- In following function we also check for collisions and whether the
    -- final IK approach is feasable:
    --displayInfo('searching for a maximum of 60 valid goal configurations...')
    local c=self:findSeveralCollisionFreeConfigsAndCheckApproach(m,300,1)

    -- Search a path from current config to a goal config. For each goal
    -- config, search 6 times a path and keep the shortest.
    -- Do this for the first 3 configs returned by findCollisionFreeConfigs.
    -- Since we do not want an approach along the negative Z axis, we place
    -- an artificial obstacle into the scene (the blue orthogon):
    if(not c) then 
        return
    end
    --sim.setThreadAutomaticSwitch(false)
    local initialApproachDirectionObstaclePose=sim.getObjectMatrix(self.approachDirectionObstacle,-1)
    sim.setObjectPosition(self.approachDirectionObstacle,theTarget,{0,0,-self.ikShift+0.01})
    sim.setObjectOrientation(self.approachDirectionObstacle,theTarget,{0,0,0})
    --sim.switchThread() -- in order see the change before next operation locks

    local txt='Found '..#c..' different goal configurations for the desired goal pose.'
    txt=txt..'&&nNow searching the shortest path of 6 searches...'
    print(txt)
    local path=self:findShortestPath(self:getConfig(),c,1)
    -- --displayInfo(nil)
    --local path = c[60]

    sim.setObjectMatrix(self.approachDirectionObstacle,-1,initialApproachDirectionObstaclePose)

  
    -- -- Follow the path:

    self:followPath(path)
    --sim.setThreadAutomaticSwitch(true)

    -- -- For the final approach, the target is the original target pose:
    -- m=sim.getObjectMatrix(theTarget,-1)

    -- -- Compute a straight-line path from current config to pose m:
    -- path=self:generateIkPath(self:getConfig(),m,self.ikSteps)

    -- -- Follow the path:
    -- self:followPath(path)

    -- --Generate a reversed path in order to move back:
    -- path=self:getReversedPath(path)

    -- -- Follow the path:
    -- self:followPath(path)

end
-- ############################################################################################################
function manipulator:getMatrixShiftedAlongZ(matrix,localZShift)
    -- Returns a pose or matrix shifted by localZShift along the matrix's z-axis
    local m={}
    for i=1,12,1 do
        m[i]=matrix[i]
    end
    m[4]=m[4]+m[3]*localZShift
    m[8]=m[8]+m[7]*localZShift
    m[12]=m[12]+m[11]*localZShift
    return m
end


function manipulator:findSeveralCollisionFreeConfigsAndCheckApproach(matrix,trialCnt,maxConfigs)
    -- Here we search for several robot configurations...
    -- 1. ..that matches the desired pose (matrix)
    -- 2. ..that does not collide in that configuration
    -- 3. ..that does not collide and that can perform the IK linear approach
    self:forbidThreadSwitches(true)
    sim.setObjectMatrix(self.ikTarget,-1,matrix)
    local cc=self:getConfig()
    local cs={}
    local l={}
    for i=1,trialCnt,1 do
        local c=self:findCollisionFreeConfigAndCheckApproach(matrix)
        if c then
            local dist=self:getConfigConfigDistance(cc,c)
            local p=0
            local same=false
            for j=1,#l,1 do
                if math.abs(l[j]-dist)<0.001 then
                    -- we might have the exact same config. Avoid that
                    same=true
                    for k=1,#self.jh,1 do
                        if math.abs(cs[j][k]-c[k])>0.01 then
                            same=false
                            break
                        end
                    end
                end
                if same then
                    break
                end
            end
            if not same then
                cs[#cs+1]=c
                l[#l+1]=dist
            end
        end
        if #l>=maxConfigs then
            break
        end
    end
    self:forbidThreadSwitches(false)
    if #cs==0 then
        cs=nil
    end
    return cs
end
function manipulator:findCollisionFreeConfigAndCheckApproach(matrix)
    -- Here we search for a robot configuration..
    -- 1. ..that matches the desired pose (matrix)
    -- 2. ..that does not collide in that configuration
    -- 3. ..that does not collide and that can perform the IK linear approach
    sim.setObjectMatrix(self.ikTarget,-1,matrix)
    -- Here we check point 1 & 2:
    local c=sim.getConfigForTipPose(self.ikGroup,self.jh,0.65,10,nil,self.collisionPairs)
    if c then
        -- Here we check point 3:
        local m=self:getMatrixShiftedAlongZ(matrix,self.ikShift)
        local path=self:generateIkPath(c,m,self.ikSteps)
        if path==nil then
            c=nil
        end
    end
    return c
end

function manipulator:forbidThreadSwitches(forbid)
    -- Allows or forbids automatic thread switches.
    -- This can be important for threaded scripts. For instance,
    -- you do not want a switch to happen while you have temporarily
    -- modified the robot configuration, since you would then see
    -- that change in the scene display.
    if forbid then
        self.forbidLevel=self.forbidLevel+1
        if self.forbidLevel==1 then
           -- sim.setThreadAutomaticSwitch(false)
        end
    else
        self.forbidLevel=self.forbidLevel-1
        if self.forbidLevel==0 then
           -- sim.setThreadAutomaticSwitch(true)
        end
    end
end

function manipulator:getConfig()
    -- Returns the current robot configuration
    local config={}
    for i=1,#self.jh,1 do
        config[i]=sim.getJointPosition(self.jh[i])
    end
    return config
end

function manipulator:setConfig(config)
    -- Applies the specified configuration to the robot
    if config then
        for i=1,#self.jh,1 do
            sim.setJointPosition(self.jh[i],config[i])
        end
    end
end

function manipulator:generateIkPath(startConfig,goalPose,steps)
    -- Generates (if possible) a linear, collision free path between a robot config and a target pose
    self:forbidThreadSwitches(true)
    local currentConfig=self:getConfig()
    self:setConfig(startConfig)
    sim.setObjectMatrix(self.ikTarget,-1,goalPose)
    local c=sim.generateIkPath(self.ikGroup,self.jh,steps,self.collisionPairs)
    self:setConfig(currentConfig)
    self:forbidThreadSwitches(false)
    return c
end

function manipulator:getConfigConfigDistance(config1,config2)
    -- Returns the distance (in configuration space) between two configurations
    local d=0
    for i=1,#self.jh,1 do
        local dx=(config1[i]-config2[i])*self.metric[i]
        d=d+dx*dx
    end
    return math.sqrt(d)
end
-- ############################################################################################################
function manipulator:findShortestPath(startConfig,goalConfigs,searchCntPerGoalConfig)
    -- This function will search for several paths between the specified start configuration,
    -- and several of the specified goal configurations. The shortest path will be returned
    self:forbidThreadSwitches(true)
    local thePath=self:findPath(startConfig,goalConfigs,searchCntPerGoalConfig)
    self:forbidThreadSwitches(false)
    return thePath
end

function manipulator:findPath(startConfig,goalConfigs,cnt)
    -- Here we do path planning between the specified start and goal configurations. We run the search cnt times,
    -- and return the shortest path, and its length
    local task=simOMPL.createTask('task')
    simOMPL.setAlgorithm(task,simOMPL.Algorithm.RRTConnect)
    local j1_space=simOMPL.createStateSpace('j1_space',simOMPL.StateSpaceType.joint_position,self.jh[1],{-170*math.pi/180},{170*math.pi/180},1)
    local j2_space=simOMPL.createStateSpace('j2_space',simOMPL.StateSpaceType.joint_position,self.jh[2],{-120*math.pi/180},{120*math.pi/180},2)
    local j3_space=simOMPL.createStateSpace('j3_space',simOMPL.StateSpaceType.joint_position,self.jh[3],{-170*math.pi/180},{170*math.pi/180},3)
    local j4_space=simOMPL.createStateSpace('j4_space',simOMPL.StateSpaceType.joint_position,self.jh[4],{-120*math.pi/180},{120*math.pi/180},0)
    local j5_space=simOMPL.createStateSpace('j5_space',simOMPL.StateSpaceType.joint_position,self.jh[5],{-170*math.pi/180},{170*math.pi/180},0)
    local j6_space=simOMPL.createStateSpace('j6_space',simOMPL.StateSpaceType.joint_position,self.jh[6],{-120*math.pi/180},{120*math.pi/180},0)
    local j7_space=simOMPL.createStateSpace('j7_space',simOMPL.StateSpaceType.joint_position,self.jh[7],{-170*math.pi/180},{170*math.pi/180},0)
    simOMPL.setStateSpace(task,{j1_space,j2_space,j3_space,j4_space,j5_space,j6_space,j7_space})
    simOMPL.setCollisionPairs(task,self.collisionPairs)
    simOMPL.setStartState(task,startConfig)
    simOMPL.setGoalState(task,goalConfigs[1])
    for i=2,#goalConfigs,1 do
        simOMPL.addGoalState(task,goalConfigs[i])
    end
    local path=nil
    local l=999999999999
    self:forbidThreadSwitches(true)
    for i=1,cnt,1 do
        local res,_path=simOMPL.compute(task,4,-1,300)
        if res and _path then
            local _l=self:getPathLength(_path)
            if _l<l then
                l=_l
                path=_path
            end
        end
    end
    self:forbidThreadSwitches(false)
    simOMPL.destroyTask(task)
    return path,l
end

function manipulator:getPathLength(path)
    -- Returns the length of the path in configuration space
    local d=0
    local l=#self.jh
    local pc=#path/l
    for i=1,pc-1,1 do
        local config1={path[(i-1)*l+1],path[(i-1)*l+2],path[(i-1)*l+3],path[(i-1)*l+4],path[(i-1)*l+5],path[(i-1)*l+6],path[(i-1)*l+7]}
        local config2={path[i*l+1],path[i*l+2],path[i*l+3],path[i*l+4],path[i*l+5],path[i*l+6],path[i*l+7]}
        d=d+self:getConfigConfigDistance(config1,config2)
    end
    return d
end

function manipulator:getConfigConfigDistance(config1,config2)
    -- Returns the distance (in configuration space) between two configurations
    local d=0
    for i=1,#self.jh,1 do
        local dx=(config1[i]-config2[i])*self.metric[i]
        d=d+dx*dx
    end
    return math.sqrt(d)
end

function manipulator:followPath(path)
    -- Follows the specified path points. Each path point is a robot configuration. Here we don't do any interpolation
    if path then
        local l=#self.jh
        local pc=#path/l
        for i=1,pc,1 do
            local config={path[(i-1)*l+1],path[(i-1)*l+2],path[(i-1)*l+3],path[(i-1)*l+4],path[(i-1)*l+5],path[(i-1)*l+6],path[(i-1)*l+7]}
            self:setConfig(config)
            --sim.switchThread()
        end
    end
end

function manipulator:getReversedPath(path)
    -- This function will simply reverse a path
    local retPath={}
    local ptCnt=#path/#self.jh
    for i=ptCnt,1,-1 do
        for j=1,#self.jh,1 do
            retPath[#retPath+1]=path[(i-1)*#self.jh+j]
        end
    end
    return retPath
end