for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
include(Pkg.dir("Knet","data","mnist.jl"))

module Lab4

using Knet,ArgParse

function predict(w,x)
    n=length(w)-2
    for i=1:2:n
        x = pool(sigm.(conv4(w[i],x;padding=0) .+ w[i+1]))
    end
    x = mat(x)
    return w[end-1]*x .+ w[end]
end

loss(w,x,ygold) = nll(predict(w,x), ygold)

lossgradient = grad(loss)

function train(w, data; lr=.15, epochs=10, iters=1800)
    for epoch=1:epochs
        for (x,y) in data
            g = lossgradient(w, x, y)
            update!(w, g, Adam())
            if (iters -= 1) <= 0
                return w
            end
        end
    end
    return w
end

function weights(;atype=KnetArray{Float32})
    w = Array{Any}(4)
    w[1] = xavier(5,5,1,3)
    w[2] = zeros(1,1,3,1)
    w[3] = xavier(10,432)
    w[4] = zeros(10,1)
    return map(a->convert(atype,a), w)
end

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="lenet.jl (c) Deniz Yuret, 2016. The LeNet model on the MNIST handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=128; help="minibatch size")
        ("--lr"; arg_type=Float64; default=0.1; help="learning rate")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--epochs"; arg_type=Int; default=3; help="number of epochs for training")
        ("--iters"; arg_type=Int; default=typemax(Int); help="maximum number of updates for training")
        ("--gcheck"; arg_type=Int; default=0; help="check N random gradients per parameter")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array and float type to use")
    end
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    println(s.description)
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    atype = eval(parse(o[:atype]))
    if atype <: Array; warn("CPU conv4 support is experimental and very slow."); end

    xtrn,ytrn,xtst,ytst = Main.mnist()
    global dtrn = minibatch(xtrn, ytrn, o[:batchsize]; xtype=atype)
    global dtst = minibatch(xtst, ytst, o[:batchsize]; xtype=atype)
    w = weights(atype=atype)
    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn,predict),:tst,accuracy(w,dtst,predict)))

    if o[:fast]
        @time (train(w, dtrn; lr=o[:lr], epochs=o[:epochs], iters=o[:iters]); gpu()>=0 && Knet.cudaDeviceSynchronize())
    else
        report(0)
        iters = o[:iters]
        for epoch=1:o[:epochs]
            @time train(w, dtrn; lr=o[:lr], epochs=1, iters=iters)
            report(epoch)
            if o[:gcheck] > 0
                gradcheck(loss, w, first(dtrn)...; gcheck=o[:gcheck], verbose=true)
            end
            if (iters -= length(dtrn)) <= 0; break; end
        end
    end
    return w
end


# This allows both non-interactive (shell command) and interactive calls like:
# $ julia lenet.jl --epochs 10
# julia> LeNet.main("--epochs 10")
PROGRAM_FILE == "lab4.jl" && main(ARGS)

end # module
