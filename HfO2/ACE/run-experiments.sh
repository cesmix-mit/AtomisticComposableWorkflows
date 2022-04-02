path=`pwd`
juliafile=fit_hfo2_ace.jl #test.jl
i=1
# RPI params: n_body, max_deg, r0, rcutof, wL, csp
for params in   'a-Hfo2-300K-NVT.extxyz 2000 2 3 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 2 4 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 2 5 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 2 6 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 3 3 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 3 4 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 3 5 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 3 6 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 4 3 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 4 4 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 4 5 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 4 6 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 5 3 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 5 4 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 5 5 1 5 1 1' \
                'a-Hfo2-300K-NVT.extxyz 2000 5 6 1 5 1 1'
do
    mkdir $path/$i; cd $path/$i;
    nohup julia ../$juliafile $params;
    i=$((i+1))
done

