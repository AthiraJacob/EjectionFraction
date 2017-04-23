require 'nn'      -- provides a normalization operator
require 'torch';   -- torch
require 'image' ;  -- for image transforms
require 'cudnn';
require 'rnn';
require 'cunn';
require 'cutorch';
-- cutorch.setDevice(3)

print '==> loading dataset'

local loader = torch.load('trainData.t7')
local loader2 = torch.load('testData.t7')

nTrain = loader.labels:size(1)
nTest = loader2.labels:size(1)

trainData = {
   data = loader.data;
   labels = loader.labels;
   size = function() return nTrain end
}

testData = {
   data = loader2.data;
   labels = loader2.labels;
   size = function() return nTest end
}



print('==> Data size:    Training   Testing')
print(nTrain)
print(nTest)