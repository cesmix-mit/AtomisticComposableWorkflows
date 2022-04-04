exp=experiments               # Experiment folder
echo "dataset,e_max_rel_error,e_mean_rel_error,e_mean_abs_error,e_rmse,"\
     "f_max_rel_error,f_mean_rel_error,f_mean_abs_error,f_rmse,"\
     "n_systems,n_params,n_body,max_deg,r0,rcutoff,wL,csp,"\
     "B_time,dB_time" >> hfo2-ace-results.csv

path=`pwd`
for currpath in $path/$exp/*/ 
do
  cat $currpath/result.csv >> hfo2-ace-results.csv
  echo "" >> hfo2-ace-results.csv
done
