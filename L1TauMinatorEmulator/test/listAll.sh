for dir in ./*/
do
   cd $dir
   pwd
   for logfile in log*.txt
   do
      tmp=${logfile#*_}
      idx=${tmp%.*}
      out=$( tail -n 1 $logfile )
      if [[ $out == "----- End Fatal Exception -------------------------------------------------" ]]; then
         echo "job num $idx not correctly finished"
         rm $PWD/Ntuple_$idx.root
         rm $PWD/log_$idx.txt
         rm $PWD/job_$idx.sh.e*
         rm $PWD/job_$idx.sh.o*
         rm $PWD/filelist_$idx.txt
      fi
   done
   cd -
done