set terminal aqua enhanced title "PO-SMC"

set datafile separator ','

filename = "exp2/TRAJ_".f.".csv"
obstacles_file = "exp2/OBSTACLE_".f.".csv"
#goal_file =  "goal-".filename
set xrange [0:15]
set yrange [0:15]
set xtics 1
set ytics 1
set grid xtics ytics
plot filename u ($2+0.5):($3 + 0.5) w l
replot obstacles_file u ($2+0.5):($3 + 0.5) w p
#replot goal_file u ($1+0.5):($2 + 0.5) w p