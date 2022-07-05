# This code will be added to PotentialLearning.jl

# Calculate metrics
function calc_metrics(x_pred, x)
    x_mae = sum(abs.(x_pred .- x)) / length(x)
    x_rmse = sqrt(sum((x_pred .- x).^2) / length(x))
    x_rsq = 1 - sum((x_pred .- x).^2) / sum((x .- mean(x)).^2)
    return x_mae, x_rmse, x_rsq
end

function get_metrics( e_train_pred, e_train, f_train_pred, f_train,
                      e_test_pred, e_test, f_test_pred, f_test,
                      B_time, dB_time, time_fitting)
    e_train_mae, e_train_rmse, e_train_rsq = calc_metrics(e_train_pred, e_train)
    f_train_mae, f_train_rmse, f_train_rsq = calc_metrics(f_train_pred, f_train)
    e_test_mae, e_test_rmse, e_test_rsq = calc_metrics(e_test_pred, e_test)
    f_test_mae, f_test_rmse, f_test_rsq = calc_metrics(f_test_pred, f_test)
    
    f_test_pred_v = collect(eachcol(reshape(f_test_pred, 3, :)))
    f_test_v = collect(eachcol(reshape(f_test, 3, :)))
    f_test_cos = dot.(f_test_v, f_test_pred_v) ./ (norm.(f_test_v) .* norm.(f_test_pred_v))
    f_test_mean_cos = mean(f_test_cos)

    metrics = OrderedDict(  "e_train_mae"      => e_train_mae,
                            "e_train_rmse"     => e_train_rmse,
                            "e_train_rsq"      => e_train_rsq,
                            "f_train_mae"      => f_train_mae,
                            "f_train_rmse"     => f_train_rmse,
                            "f_train_rsq"      => f_train_rsq,
                            "e_test_mae"       => e_test_mae,
                            "e_test_rmse"      => e_test_rmse,
                            "e_test_rsq"       => e_test_rsq,
                            "f_test_mae"       => f_test_mae,
                            "f_test_rmse"      => f_test_rmse,
                            "f_test_rsq"       => f_test_rsq,
                            "f_test_mean_cos"  => f_test_mean_cos,
                            "B_time [s]"       => B_time,
                            "dB_time [s]"      => dB_time,
                            "time_fitting [s]" => time_fitting)
    return metrics
end

# Plot variables
function plot_energy(e_pred, e_true)
    r0 = minimum(e_true); r1 = maximum(e_true); rs = (r1-r0)/10
    plot( e_true, e_pred, seriestype = :scatter, markerstrokewidth=0,
          label="", xlabel = "E DFT | eV/atom", ylabel = "E predicted | eV/atom")
    p = plot!( r0:rs:r1, r0:rs:r1, label="")
    return p
end

function plot_forces(f_pred, f_true)
    f_pred_v = collect(eachcol(reshape(f_pred, 3, :)))
    f_true_v = collect(eachcol(reshape(f_true, 3, :)))
    r0 = 0; r1 = ceil(maximum(norm.(f_true_v)))
    plot( norm.(f_true_v), norm.(f_pred_v), seriestype = :scatter, markerstrokewidth=0,
          label="", xlabel = "|F| DFT | eV/Å", ylabel = "|F| predicted | eV/Å", 
          xlims = (r0, r1), ylims = (r0, r1))
    p = plot!( r0:r1, r0:r1, label="")
    return p
end

function plot_cos(f_pred, f_true)
    f_pred_v = collect(eachcol(reshape(f_pred, 3, :)))
    f_true_v = collect(eachcol(reshape(f_true, 3, :)))
    f_cos = dot.(f_true_v, f_pred_v) ./ (norm.(f_true_v) .* norm.(f_pred_v))
    p = plot( f_cos, seriestype = :scatter, markerstrokewidth=0,
              label="", xlabel = "F DFT vs F predicted", ylabel = "cos(α)")
    return p
end

