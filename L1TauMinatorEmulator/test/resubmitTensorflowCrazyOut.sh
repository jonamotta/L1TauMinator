user=$( pwd | cut -d"/" -f4)

for logfile in log*.txt
do
   tmp=${logfile#*_}
   idx=${tmp%.*}
   out=$( grep "Tensorflow inference went ballistic" $logfile )
   if [[ $out == "** ERROR: the Tensorflow inference went ballistic in this job! Please re-run it!" ]]; then
      echo "The Tensorflow inference went ballistic in job $idx - resubmitting it"
      /home/llr/cms/$user/t3submit -short job_$idx.sh
   fi
done
