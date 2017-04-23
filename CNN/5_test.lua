----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   score = 0

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size(),opt.batchSize do

      xlua.progress(t, testData:size())

      local testInputs = testData.data[{{t,math.min(t+opt.batchSize-1,testData:size())}}]:cuda()
      local testLabels = testData.labels[{{t,math.min(t+opt.batchSize-1,testData:size())}}]:cuda()

      outputs = model:forward(testInputs)

      for k = 1,testInputs:size(1) do
         z = outputs[k] - testLabels[k]
         score = score + z*z
      end

   end
   print(testData:size())
   score = score/testData:size()

   if score < bestValidScore then
      local filename = paths.concat(opt.save, 'model.net')
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('\n==> saving model to '..filename)
      torch.save(filename, model)
      bestValidScore = score      
   end


   print("\nValidation MSE score: "..(score))

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = score}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   -- confusion:zero()
end
