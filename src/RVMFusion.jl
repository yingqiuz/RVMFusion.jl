#  RVM for lower quality data
module RVMFusion
using Statistics, LinearAlgebra, StatsBase, Distributions
using ThreadsX, Transducers, Folds, ProgressMeter
import LoopVectorization

export RVM, RVM!, sigmoid

include("utils.jl")
include("binomial.jl")
include("multinomial.jl")

end
