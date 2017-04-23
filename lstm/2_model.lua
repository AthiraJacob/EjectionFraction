require 'torch';   -- torch
require 'image' ;  -- for image transforms
require 'nn' ;     -- provides all sorts of trainable modules/layers
require 'cudnn';
require 'rnn';
require 'cutorch';
require 'cunn'


print '==> define parameters'


inputSize =  55000
hiddenSize1 = 1000
hiddenSize2 = 500
hiddenSize3 = 100
outputSize = 1
rho = 15

dropout = 0.5

-- model = nn.Sequential()
-- model:add(nn.Sequencer(nn.Linear(inputSize, 10000)))
-- -- model:add(nn.Sequencer(nn.Dropout(dropout)))
-- model:add(nn.Sequencer(nn.Linear(10000 , hiddenSize1)))
-- -- model:add(nn.Sequencer(nn.Dropout(dropout)))
-- model:add(nn.Sequencer(nn.FastLSTM(hiddenSize1, hiddenSize2, rho)))
-- -- model:add(nn.Sequencer(nn.Dropout(dropout)))
-- model:add(nn.Sequencer(nn.FastLSTM(hiddenSize2, hiddenSize3, rho)))
-- -- model:add(nn.Sequencer(nn.Dropout(dropout)))
-- -- model:add(nn.Sequencer(nn.FastLSTM(hiddenSize3, hiddenSize3, rho)))
-- model:add(nn.SelectTable(-1))
-- -- model:add(nn.Dropout(dropout))
-- model:add(nn.Linear(hiddenSize3, 25))
-- -- model:add(nn.Dropout(dropout))
-- model:add(nn.Linear(25, outputSize))
-- model:add(cudnn.Sigmoid())

-- model = require('weight-init')(model, 'heuristic')

model = nn.Sequential()
model:add(nn.Sequencer(nn.Linear(6875, 4000)))
model:add(nn.Sequencer(nn.ReLU()))
-- model:add(nn.Sequencer(nn.Dropout(dropout)))
-- model:add(nn.Sequencer(nn.Linear(3000 , 1000)))
model:add(nn.Sequencer(nn.Dropout(dropout)))
model:add(nn.Sequencer(nn.FastLSTM(4000, 2000, rho)))
model:add(nn.Sequencer(nn.ReLU()))
model:add(nn.Sequencer(nn.Dropout(dropout)))
model:add(nn.Sequencer(nn.FastLSTM(2000, 1000, rho)))
model:add(nn.SelectTable(-1))
model:add(nn.Dropout(dropout))
model:add(nn.Linear(1000, 100))
model:add(nn.ReLU())
model:add(nn.Dropout(dropout))
model:add(nn.Linear(100, outputSize))
model:add(cudnn.Sigmoid())


model:cuda()

print('==>Model')
print(model)




-- model = nn.Sequential()
-- model:add(nn.Linear(3,5))
-- model:add(nn.Linear(5,2))
-- for i = 1,#model do
-- 	print(i)
-- 	model:get(i):getParameters():clamp(-0.5,0.5)
-- end
