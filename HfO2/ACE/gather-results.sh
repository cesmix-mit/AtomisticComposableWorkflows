# Result file
results=hfo2-ace-results.csv
# Experiment folder
exp=experiments

# Result header
echo "dataset,"\
     "n_systems,n_params,n_body,max_deg,r0,rcutoff,wL,csp," \
     "e_train_rmse,e_train_mae,e_train_mre,e_train_maxre," \
     "f_train_rmse,f_train_mae,f_train_mre,f_train_maxre," \
     "e_test_rmse,e_test_mae,e_test_mre,e_test_maxre," \
     "f_test_rmse,f_test_mae,f_test_mre,f_test_maxre,"\
     "B_time,dB_time" >> $results

# Gather results
path=`pwd`
for currpath in $path/$exp/*/ 
do
  cat $currpath/results.csv >> $results
  echo "" >> $results
done

