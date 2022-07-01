# This section will feed PotentialLearning.jl?

# Calculate metrics
function calc_metrics(x_pred, x)
    x_mae = sum(abs.(x_pred .- x)) / length(x)
    x_rmse = sqrt(sum((x_pred .- x).^2) / length(x))
    x_rsq = 1 - sum((x_pred .- x).^2) / sum((x .- mean(x)).^2)
    return x_mae, x_rmse, x_rsq
end

# Post-process output: calculate metrics, save results and plots
function postproc( input, e_train_pred, e_train, f_train_pred, f_train,
                   e_test_pred, e_test, f_test_pred, f_test, 
                   B_time, n_params, dB_time, time_fitting)

    # Calculate metrics
    e_train_mae, e_train_rmse, e_train_rsq = calc_metrics(e_train_pred, e_train)
    f_train_mae, f_train_rmse, f_train_rsq = calc_metrics(f_train_pred, f_train)
    e_test_mae, e_test_rmse, e_test_rsq = calc_metrics(e_test_pred, e_test)
    f_test_mae, f_test_rmse, f_test_rsq = calc_metrics(f_test_pred, f_test)

    f_test_pred_v = collect(eachcol(reshape(f_test_pred, 3, :)))
    f_test_v = collect(eachcol(reshape(f_test, 3, :)))
    f_test_cos = dot.(f_test_v, f_test_pred_v) ./ (norm.(f_test_v) .* norm.(f_test_pred_v))
    f_test_mean_cos = mean(f_test_cos)
    

    # Save results
    dataset_filename = input["dataset_filename"]
    n_train_sys = input["n_train_sys"]
    n_test_sys = input["n_test_sys"]
    n_body = input["n_body"]
    max_deg = input["max_deg"]
    r0 = input["r0"]
    rcutoff = input["rcutoff"]
    wL = input["wL"]
    csp = input["csp"]
    w_e = input["w_e"]
    w_f = input["w_f"]
    write(path*"results.csv", "dataset,\
                          n_train_sys,n_test_sys,n_params,n_body,max_deg,r0,\
                          rcutoff,wL,csp,w_e,w_f,\
                          e_train_mae,e_train_rmse,e_train_rsq,\
                          f_train_mae,f_train_rmse,f_train_rsq,\
                          e_test_mae,e_test_rmse,e_test_rsq,\
                          f_test_mae,f_test_rmse,f_test_rsq,\
                          f_test_mean_cos,B_time,dB_time,time_fitting
                          $(dataset_filename), \
                          $(n_train_sys),$(n_test_sys),$(n_params),$(n_body),$(max_deg),$(r0),\
                          $(rcutoff),$(wL),$(csp),$(w_e),$(w_e),\
                          $(e_train_mae),$(e_train_rmse),$(e_train_rsq),\
                          $(f_train_mae),$(f_train_rmse),$(f_train_rsq),\
                          $(e_test_mae),$(e_test_rmse),$(e_test_rsq),\
                          $(f_test_mae),$(f_test_rmse),$(f_test_rsq),\
                          $(f_test_mean_cos),$(B_time),$(dB_time),$(time_fitting)")

    write(path*"results-short.csv", "dataset,\
                          n_train_sys,n_test_sys,n_params,n_body,max_deg,r0,rcutoff,\
                          e_test_mae,e_test_rmse,\
                          f_test_mae,f_test_rmse,\
                          f_test_mean_cos,\
                          B_time,dB_time,time_fitting
                          $(dataset_filename),\
                          $(n_train_sys),$(n_test_sys),$(n_params),$(n_body),$(max_deg),$(r0),$(rcutoff),\
                          $(e_test_mae),$(e_test_rmse),\
                          $(f_test_mae),$(f_test_rmse),\
                          $(B_time),$(dB_time),$(time_fitting)")

    # Save plots
    r0 = minimum(e_test); r1 = maximum(e_test); rs = (r1-r0)/10
    plot( e_test, e_test_pred, seriestype = :scatter, markerstrokewidth=0,
          label="", xlabel = "E DFT | eV/atom", ylabel = "E predicted | eV/atom")
    plot!( r0:rs:r1, r0:rs:r1, label="")
    savefig(path*"e_test.png")

    r0 = 0; r1 = ceil(maximum(norm.(f_test_v)))
    plot( norm.(f_test_v), norm.(f_test_pred_v), seriestype = :scatter, markerstrokewidth=0,
          label="", xlabel = "|F| DFT | eV/Å", ylabel = "|F| predicted | eV/Å", 
          xlims = (r0, r1), ylims = (r0, r1))
    plot!( r0:r1, r0:r1, label="")
    savefig(path*"f_test.png")

    plot( f_test_cos, seriestype = :scatter, markerstrokewidth=0,
          label="", xlabel = "F DFT vs F predicted", ylabel = "cos(α)")
    savefig(path*"f_test_cos.png")
end
