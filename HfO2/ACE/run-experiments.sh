exp=experiments               # Experiment folder
exppars=experiment-params.dat # Experiment parameter file
juliafile=fit-hfo2-ace.jl     # Fitting program

path=`pwd`
mkdir $path/$exp
{
    read
    while IFS= read -r params
    do
        expi=`echo "$params" | tr ' ' -`
        mkdir $path/$exp/$expi; cd $path/$exp/$expi;
        echo "$params" > result.csv
        #nohup julia ../$juliafile $params &
    done 
} < "$exppars"
