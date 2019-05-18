set terminal aqua enhanced title "PO-SMC"

set datafile separator ','

#filename = "map-14-4-1.txt"
#obstacles_file = "obstacles-".filename
#goal_file =  "goal-".filename
set xrange [0:15]
set yrange [0:15]
set xtics 1
set ytics 1
set grid xtics ytics
plot filename u ($2+0.5):($3 + 0.5) w l
#replot obstacles_file u ($1+0.5):($2 + 0.5) w p
#replot goal_file u ($1+0.5):($2 + 0.5) w p