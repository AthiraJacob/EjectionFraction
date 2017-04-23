----------------------------------------------------------------------
-- This script demonstrates how to define a training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It shows how to:
--   + construct mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to several optmization
--     methods: SGD, L-BFGS.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0, 'momentum (SGD only)')
   cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
   cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- CUDA?
if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

----------------------------------------------------------------------
print '==> defining some tools'

-- classes
-- classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
-- confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd

else
   error('unknown optimization method')
end

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

  -- do one epoch
  -- print('==> doing epoch on training data:')
  print("\n\n==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  score = 0
  print(trainData:size())

  for t = 1,trainData:size(),opt.batchSize do
  -- disp progress
    xlua.progress(t, trainData:size())

    local inputs = trainData.data[{{t,math.min(t+opt.batchSize-1,trainData:size())}}]:cuda()
    local targets = trainData.labels[{{t,math.min(t+opt.batchSize-1,trainData:size())}}]:cuda()

    local feval = function(x)
               -- get new parameters
      if x ~= parameters then
          parameters:copy(x)
      end

      -- reset gradients
      gradParameters:zero()

      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs,targets)

      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
         -- locals:
         local norm,sign= torch.norm,torch.sign

         -- Loss:

         f = f + opt.coefL1 * norm(parameters,1)
         f = f + opt.coefL2 * norm(parameters,2)^2/2

         -- Gradients:
         gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
      end

      for k = 1,inputs:size(1) do
        score = score + (outputs[k] - targets[k])*(outputs[k] - targets[k])
      end


      return f,gradParameters
    end

    -- optimize on current mini-batch
    if optimMethod == optim.asgd then
      _,_,average = optimMethod(feval, parameters, optimState)
    elseif optimMethod == optim.rmsprop2 then
      _,fs,alr = optimMethod(feval, parameters, optimState,epoch)
    else
      _,fs = optimMethod(feval,parameters,optimState)
    end

    -- err = err + fs[1]

   end

   if optimMethod == optim.rmsprop2 then
      print('\nLearningRate: ', alr)
    end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   -- print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- err = err / trainData:size()
   score = score / trainData:size()

  local filename = paths.concat(opt.save, 'latestModel.net')
  os.execute('mkdir -p ' .. sys.dirname(filename))
  print('\n==> saving model to '..filename)
  torch.save(filename, model)
  bestTrainScore = score

  print("\n==> MSE Error: "..(score))

  trainLogger:add{['% mean class accuracy (train set)'] = err}
  if opt.plot then
    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    trainLogger:plot()
  end
  epoch = epoch + 1
end