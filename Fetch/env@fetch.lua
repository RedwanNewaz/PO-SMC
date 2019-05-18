env = {
    project_dir = '/Users/redwannewaz/Projects/research@quadrotor/PO-SMC/pomdpy/config/',
    obstacles = {},
    goal = {},
    start = {},
    caution = {},
    dim = {},
    dx = 0,
    dy = 0
}
function env:new(o,filename)
    o = o or {}
    setmetatable(o,self)
    self.__index = self
    self.__file = self.project_dir..filename
    local yellow={1.,1.0,0.0}
    local red={1.,0.0,0.0}
	lineSize=5 -- in points
	maximumLines=9999
    self.drawingContainer=simAddDrawingObject(sim_drawing_lines,lineSize,0,-1,maximumLines,yellow) -- adds a line
    self.drawingRed=simAddDrawingObject(sim_drawing_lines,lineSize,0,-1,maximumLines,red) -- adds a line

    return o

end
function env:file_exists(file)
    local f = io.open(file, "rb")
    if f then f:close() end
    return f ~= nil
end
 
function env:read_map()
    --[[
        read txt file and draw different objects on the floor
        require mapping from grid to floor
    ]]
    file = self.__file
    if not self:file_exists(file) then return {} end
    local lines = {}
    
    local j=0

    for line in io.lines(file) do
        if j>0 then 
         
            for i = 1, #line do
                local c = line:sub(i,i)
                if (c=='X')then 
                    -- print(c,i,j)
                    table.insert(self.obstacles,{i,j})
                elseif (c=='G')then 
                    table.insert(self.goal,{i,j})
                elseif (c=='S')then 
                    table.insert(self.start,{i,j})
                elseif (c=='C')then 
                    table.insert(self.caution,{i,j})
                end  
            
            end
            lines[#lines + 1] = line
        else
            for token in string.gmatch(line, "[^%s]+") do
                table.insert(self.dim,tonumber(token))
             end
            
        end
        j=j+1
    end
    print(self.dim)
    print('start',self.start,'obstacle',self.obstacles,'goal',self.goal,'caution',self.caution)
    return lines
end

function env:transform(box_val)    
    local map_size = self.dim 
    x_coord,self.dx = self:map({box_val[1],box_val[2]},{box_val[3],map_size[1]})
    y_coord,self.dy = self:map({box_val[1],box_val[4]},{box_val[3],map_size[2]})
    local XY = {}
    for x=1,#x_coord-1 do 
        for y=1,#y_coord-1 do 
            table.insert(XY,{x_coord[x]+self.dx/2,y_coord[y]+self.dy/2})
        end
    end
    return XY
end


function env:map(x,y)
    dx = math.abs(x[2]-x[1])
    dy = math.abs(y[2]-y[1])
    scale = dx/dy
    local r ={}
    for i = x[1],x[2],scale do 
        table.insert(r,i)
    end
    return r,scale
end

function env:create_shape(map_position,primitiveType,color)
    --[[
    primitiveType: 0 for a cuboid, 1 for a sphere, 2 for a cylinder and 3 for a cone
    options: Bit-coded: if bit0 is set (1), backfaces are culled. If bit1 is set (2), edges are visible. If bit2 is set (4), the shape appears smooth. If bit3 is set (8), the shape is respondable. 
            If bit4 is set (16), the shape is static. If bit5 is set (32), the cylinder has open ends
    sizes: 3 values indicating the size of the shape
    mass: the mass of the shape
    precision: 2 values that allow specifying the number of sides and faces of a cylinder or sphere. Can be NULL for default values
    --]]
        -- primitiveType =0
        local options = 16
        local sizes = {.81,.81,1}
        local mass = 1
        local precision=nil
        table.insert(map_position,0) -- add z axis to coordinate 
        local objectHandle=sim.createPureShape(primitiveType,
                    options,sizes,mass, precision)
        sim.setObjectPosition(objectHandle,-1,map_position)
        sim.setShapeColor(objectHandle,"",sim_colorcomponent_ambient,color)
    end
--[[
    Grid lines on the floor
]]
function env:DrawGrid(box_val,map_size)
     
    
    x_coord,dx = self:map({box_val[1],box_val[2]},{box_val[3],map_size[1]})
    y_coord,dy = self:map({box_val[1],box_val[4]},{box_val[3],map_size[2]})

    
    local XY = self:GetCoord(x_coord,y_coord,map_size,'y');
    self:drawVerticalLines(XY)
    local XY = self:GetCoord(x_coord,y_coord,map_size,'x')
    --print('XY',XY)
    self:drawHorizontalLines(XY)
end

function env:GetCoord(x_coord,y_coord,map_size, axis)

    local diffX = 1;
    local diffY = 1;
    if axis=='x' then
       diffX = map_size[1];
    else
       diffY = map_size[2];
    end

    local XY={};
    local nX = #x_coord;
    local nY = #y_coord;
    for x=1,nX,diffX do
        for y=1,nY,diffY do
            table.insert(XY,{x_coord[x],y_coord[y]})
        end
    end
    return XY
end 
function env:drawEdge(pt0,pt1)
    local data={pt0[1],pt0[2],0.01,pt1[1],pt1[2],0.01}
    simAddDrawingObjectItem(self.drawingContainer,data)
end

function env:drawEdgeRed(pt0,pt1)
    local data={pt0[1],pt0[2],0.01,pt1[1],pt1[2],0.01}
    simAddDrawingObjectItem(self.drawingRed,data)
end

function env:drawHorizontalLines(X)
    local n = math.floor(#X/2);
    --print(#X,n)
    for i=1,n do
        --print(X[i],X[n+i])
       self:drawEdge(X[i],X[n+i])
    end
end
function env:drawVerticalLines(X)

    for i=1,#X,2 do
       self:drawEdge(X[i],X[i+1])
    end
end

function env:drawX(index,box_val,map_size)
    x_coord,dx = self:map({box_val[1],box_val[2]},{box_val[3],map_size[1]})
    y_coord,dy = self:map({box_val[1],box_val[4]},{box_val[3],map_size[2]})
    local XY={};

    for x=1,#x_coord-1 do
        for y=1,#y_coord-1 do
            table.insert(XY,{x_coord[x]+dx/2,y_coord[y]+dy/2});
        end
    end

    
    local x= XY[index][1]
    local y= XY[index][2]
    print(x,y)
    cross1 ={
        {x-dx/4,y+dy/2},
        {x-dx/2,y-dy/2}    
    }
    cross2 ={
        {x-dx/4,y-dy/2},
        {x-dx/2,y+dy/2}   
    }
    print(cross1,cross2)   
    self:drawEdgeRed(cross1[1],cross1[2])
    self:drawEdgeRed(cross2[1],cross2[2])

end

