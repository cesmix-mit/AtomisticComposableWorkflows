# Result file
results=results.csv
# Experiment folder
exp=experiments

# Result header
echo "dataset,"\
     "n_systems,n_params,n_body,max_deg,r0,"\
     "rcutoff,wL,csp,e_weight,f_weight,"\
     "e_train_mae,e_train_mre,e_train_rmse,e_train_rsq,"\
     "f_train_mae,f_train_mre,f_train_rmse,f_train_rsq,"\
     "e_test_mae,e_test_mre,e_test_rmse,e_test_rsq,"\
     "f_test_mae,f_test_mre,f_test_rmse,f_test_rsq,"\
     "B_time,dB_time,time_fitting"  >> $exp/$results

# Gather results
path=`pwd`
for currpath in $path/$exp/*/ 
do
  tail -n 1 $currpath/results.csv >> $exp/$results
  echo "" >> $exp/$results
done

