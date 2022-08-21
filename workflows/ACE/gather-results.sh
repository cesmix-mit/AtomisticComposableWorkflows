# Result file
results=results.csv

# Experiment folder
exp=experiments

# Current path
path=`pwd`

# Header
filename=`ls $path/$exp | tail -n1` 
cat $path/$exp/$filename/input.csv | cut -d, -f1 | paste -s -d, | tr -d '\n' >> $exp/$results
echo "," | tr -d '\n' >> $exp/$results
cat $path/$exp/$filename/metrics.csv | cut -d, -f1 | paste -s -d, | tr -d '\n' >> $exp/$results
echo "" >> $exp/$results

# Content
for currpath in $path/$exp/*/ 
do
    cat $currpath/input.csv | cut -d, -f2 | paste -s -d, | tr -d '\n' >> $exp/$results
    echo "," | tr -d '\n' >> $exp/$results
    cat $currpath/metrics.csv | cut -d, -f2 | paste -s -d, | tr -d '\n' >> $exp/$results
    echo "" >> $exp/$results
done

