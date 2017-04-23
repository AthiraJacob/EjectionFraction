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
require 'cudnn'
require 'cunn'
require 'cutorch';
-- cutorch.setDevice(3)
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
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS | RMS')
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0, 'momentum (SGD only)')
   cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
   cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
   cmd:option('-coefL1',0.00005,'L1 norm')
   cmd:option('-coefL2',0,'L2 norm')
   cmd:option('-gradClip',0,'Clip gradients? 0|1')
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

elseif opt.optimization == 'RMS' then
	optimState = {
      learningRate = opt.learningRate,
   }
   optimMethod = optim.rmsprop

elseif opt.optimization == 'adam' then
  optimState = {
      learningRate = opt.learningRate,
   }
   optimMethod = optim.adam

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

  if opt.gradClip ==1 then
    opt.coefL1 = 0
    opt.coefL2 = 0
    for i = 1,#model do
      model:get(i):getParameters():clamp(-0.5,0.5)
    end
  end
     
  for t = 1,trainData:size(),opt.batchSize do
  -- disp progress
    xlua.progress(t, trainData:size())
    input_batch = {}
    temp = math.min(t+opt.batchSize-1,trainData:size()) 
    inputSize = trainData.data[t][1]:size()[1];

    for i = 1,rho do
    	table.insert(input_batch,torch.Tensor(temp-t+1,inputSize):zero():cuda())
    end

    k = 1;
    for i = t,temp do
      for j = 1,rho do
      	input_batch[j][k] = trainData.data[i][j]:cuda()
      end
      	k = k+1
    end

    -- print(input_batch)
    target_batch = trainData.labels[{{t,math.min(t+opt.batchSize-1,trainData:size())}}]:cuda()
    -- print(target_batch)
    local feval = function(x)
      if x ~= parameters then
          parameters:copy(x)
      end

      gradParameters:zero()
      
      local prediction = model:forward(input_batch)
      local err = criterion:forward(prediction, target_batch)
      local gradOutputs = criterion:backward(prediction, target_batch)
      model:backward(input_batch, gradOutputs)
      score = score + (target_batch - prediction):dot(target_batch - prediction)
  
     
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
                          -- locals:
        local norm,sign= torch.norm,torch.sign
                          -- Loss:
        err = err + opt.coefL1 * norm(parameters,1)
        err = err + opt.coefL2 * norm(parameters,2)^2/2
        -- Gradients:
        gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
      end
      -- model:updateParameters(opt.learningRate)
      return err,gradParameters
    end
     -- optimize on current mini-batch
    if optimMethod == optim.asgd then
      _,_,average = optimMethod(feval, parameters, optimState)
    elseif optimMethod == optim.rmsprop  then
      _,fs,alr = optimMethod(feval, parameters, optimState)
    else
      _,fs = optimMethod(feval,parameters,optimState)
    end

  end
 

  score = score / trainData:size()
  
   --time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

 local filename = paths.concat(opt.save, 'latestModel.net')
  os.execute('mkdir -p ' .. sys.dirname(filename))
  print('\n==> saving model to '..filename)
  torch.save(filename, model)
  
  trainLogger:add{['MSE error = '] = 'Epoch '..epoch..' '..score}

  print("\n==> MSE Error: "..(score))

  epoch = epoch + 1

end


