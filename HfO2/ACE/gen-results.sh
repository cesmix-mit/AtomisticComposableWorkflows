path=`pwd`
echo "dataset,e_max_rel_error,e_mean_rel_error,e_mean_abs_error,e_rmse,f_max_rel_error,f_mean_rel_error,f_mean_abs_error,f_rmse,n_systems,nparams,n_body,max_deg,r0,rcutoff,wL,csp,B_time,dB_time" >> hfo2-ace-results.dat
for i in {1..16}
do
  cat $path/$i/result.dat >> hfo2-ace-results.dat
  echo "" >> hfo2-ace-results.dat
done
