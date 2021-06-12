#  RVM for lower quality data
module RVMFusion
using Statistics, LinearAlgebra, StatsBase, Distributions
using ThreadsX, Transducers, Folds, ProgressMeter
import LoopVectorization

export RVM, RVM!

include("utils.jl")
include("binary.jl")
include("multiclass.jl")

end
