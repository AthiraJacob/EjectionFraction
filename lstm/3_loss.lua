require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions
require 'cudnn'
require 'rnn'



--model:add(cudnn.Sigmoid())
criterion = nn.MSECriterion()
model:cuda()
-- criterion.sizeAverage = false
criterion:cuda()

print '==> here is the loss function:'
print(criterion)

