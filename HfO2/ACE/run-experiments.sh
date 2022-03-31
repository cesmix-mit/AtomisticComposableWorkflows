path=`pwd`
juliafile=fit_hfo2_ace.jl #test.jl
i=1
# RPI params: n_body, max_deg, r0, rcutof, wL, csp
for params in '4 6 1 5 1 1' \
              '4 7 1 5 1 1' \
              '5 6 1 5 1 1' \
              '6 6 1 5 1 1' \
              '5 6 1 7 1 1'
do
    mkdir $path/$i; cd $path/$i;
    nohup julia ../$juliafile $params;
    i=$((i+1))
done

