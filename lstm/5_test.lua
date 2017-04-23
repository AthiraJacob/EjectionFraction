-- ----------------------------------------------------------------------
-- -- This script implements a test procedure, to report accuracy
-- -- on the test data. Nothing fancy here...
-- --
-- -- Clement Farabet
-- ----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'cutorch';
-- cutorch.setDevice(3)
-- ----------------------------------------------------------------------
-- print '==> defining test procedure'

-- -- test function
-- function test()
--    -- local vars
--    local time = sys.clock()

--    -- averaged param use?
--    if average then
--       cachedparams = parameters:clone()
--       parameters:copy(average)
--    end

--    score = 0

--    -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
--    model:evaluate()

--    -- test over test data
--    print('==> testing on test set:')
--    for t = 1,testData:size(),opt.batchSize do

--       xlua.progress(t, testData:size())

--       local testInputs = testData.data[{{t,math.min(t+opt.batchSize-1,testData:size())}}]:cuda()
--       local testLabels = testData.labels[{{t,math.min(t+opt.batchSize-1,testData:size())}}]:cuda()

--       outputs = model:forward(testInputs)

--       for k = 1,testInputs:size(1) do
--          z = outputs[k] - testLabels[k]
--          score = score + z*z
--       end

--    end
--    print(testData:size())
--    score = score/testData:size()

--    if score < bestScore then
--       local filename = paths.concat(opt.save, 'model.net')
--       os.execute('mkdir -p ' .. sys.dirname(filename))
--       print('\n==> saving model to '..filename)
--       torch.save(filename, model)
--       bestScore = score      
--    end


--    print("\nValidation MSE score: "..(score))

--    -- timing
--    time = sys.clock() - time
--    time = time / testData:size()
--    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

--    -- print confusion matrix

--    -- update log/plot
--    testLogger:add{['% mean class accuracy (test set)'] = score}
--    if opt.plot then
--       testLogger:style{['% mean class accuracy (test set)'] = '-'}
--       testLogger:plot()
--    end

--    -- averaged param use?
--    if average then
--       -- restore parameters
--       parameters:copy(cachedparams)
--    end
   
--    -- next iteration:
--    -- confusion:zero()
-- end




function test()


   -- local vars
   local time = sys.clock()
   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end
   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

  -- test over test data
   print('==> testing on test set:')

  print(testData:size())

  score_test = 0
  

  for t = 1,testData:size(1),opt.batchSize do
    
  -- disp progress
    xlua.progress(t, testData:size())
    
    input_batch = {}
    temp = math.min(t+opt.batchSize-1,testData:size()) 
    inputSize = testData.data[t][1]:size()[1];

    for i = 1,rho do
      table.insert(input_batch,torch.Tensor(temp-t+1,inputSize):zero():cuda())
    end

    k = 1;
    for i = t,temp do
      for j = 1,rho do
        input_batch[j][k] = testData.data[i][j]
      end
        k = k+1
    end

    target_batch = testData.labels[{{t,temp}}]:cuda()
    
    local prediction = model:forward(input_batch)
    score_test = score_test + (target_batch - prediction):dot(target_batch - prediction)
      
  end
  
  score_test = score_test / testData:size()

  -- print(score_test)
  temp2 = epoch-1
  testLogger:add{['Validation error = '] = 'Epoch '..temp2..' '..score_test}

  if score_test < bestScore then
      local filename = paths.concat(opt.save, 'model.net')
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('\n==> saving model to '..filename)
      torch.save(filename, model)
      bestScore = score_test      
   end


   print("\nValidation MSE score: "..(score_test))

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')


end