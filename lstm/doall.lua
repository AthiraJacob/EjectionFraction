
require 'dp'
require 'rnn'

require 'torch'

----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- -- data:
-- cmd:option('-size', 'full', 'how many samples do we load: small | full | extra')
-- -- model:
-- cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
-- -- loss:
-- cmd:option('-loss', 'mse', 'type of loss function to minimize: nll | mse | margin')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'cuda', 'type: double | float | cuda')
cmd:option('-visualize', false, 'visualize input data and weights during training')
cmd:option('-coefL1',0,'coefL1')
cmd:option('-coefL2',0,'coefL2')
cmd:option('-gradClip',0,'Clip gradients? 0|1')

cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'

dofile '1_data.lua'
dofile '2_model.lua'
dofile '3_loss.lua'
dofile '4_train.lua'
dofile '5_test.lua'

----------------------------------------------------------------------
print '==> training!'

bestScore = 1000000
i = 0
while i<1000 do
	print '\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
   train()
   test()
   i = i+1
   print '\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
end
