-- check results of lstm

require 'torch';   -- torch
require 'image';   -- for image transforms
require 'nn';      -- provides all sorts of trainable modules/layers
require 'cudnn';
require 'rnn';
require 'cunn'
require 'gnuplot';

tr = 1
te = 1

py = require('fb.python');
mod = torch.load('results/model.net')
-- mod = torch.load('../threadedLSTM/results/model.net')

-- new data
-- Load volumes
-- print('==> Loading labels..')
-- py.exec([=[
-- import numpy as np
-- from openpyxl import load_workbook
-- wb = load_workbook(filename = '../data.xlsx',data_only = 1)
-- sh = wb['Sheet']
-- sysVolumes = np.zeros(700)
-- diaVolumes = np.zeros(700)
-- for i in range(700):
-- 	sysVolumes[i] = sh['O'+str(i+2)].value 
-- 	diaVolumes[i] = sh['P'+str(i+2)].value
-- 	]=])
-- -- R and S
-- sysVolumes = py.eval('sysVolumes')
-- diaVolumes = py.eval('diaVolumes')

-- test = torch.load('testData.t7')

if tr == 1  then
test = torch.load('trainData.t7')
sum_err = torch.Tensor(2):zero()
trNo = #test.data

start = 1
last = 604
err_list = torch.Tensor(trNo,2)
ef_err = torch.Tensor(trNo)
efList = torch.Tensor(trNo,2)


mean = torch.load('dataMean.t7')
std = torch.load('dataStd.t7')
k = 1
print('==> Testing with training data..')
for pnum = 1,trNo  do
	
	-- print(pnum)
	testData = {
		data = test.data[pnum];
		labels = test.labels[{{pnum}}];
		size = function() return 1 end;
	}

	for t = 1, #testData.data do
		-- testData.data[t] = (testData.data[t]:mul(std)):add(mean)
		testData.data[t] = testData.data[t]:cuda()
	end

	out = mod:forward(testData.data);
	ef_True = testData.labels:float()
	efList[{pnum,{1}}] = ef_True
	out = out:float()
	efList[{pnum,{2}}] = out
	ef_err[{{pnum}}] = ef_True - out
	k = k+1
end
-- 	print(out[1]*500 ..' '.. out[2]*600)
-- print(testData.labels[{1,1}]*500 ..' '.. testData.labels[{1,2}]*600)
	-- sysCalc = out[1]*500
	-- diaCalc = out[2]*600
	-- sysTrue = testData.labels[{1,1}]*500
	-- diaTrue = testData.labels[{1,2}]*600

	-- err = torch.Tensor(2)
	-- err[1] = torch.abs(sysTrue-sysCalc)
	-- err[2] = torch.abs(diaTrue-diaCalc)
	
	-- efTrue = (diaTrue-sysTrue)/diaTrue
	-- efCalc = (diaCalc-sysCalc)/diaCalc

	-- err_list[{pnum-start+1}] = err 
	-- ef_err[{pnum-start+1}] = efTrue-efCalc
	-- sum_err = sum_err+err



mt = require('fb.mattorch')
mt.save('ef_train.mat',efList)
end

-- sum_err = sum_err/(last-start+1)
-- sys = err_list[{{1,97},{1}}]
-- dia = err_list[{{1,97},{2}}]
if te == 1  then

test = torch.load('testData.t7')
sum_err = torch.Tensor(2):zero()
teNo = #test.data

start = 605
last = 700
err_list = torch.Tensor(teNo,2):zero()
ef_err = torch.Tensor(teNo):zero()
efList = torch.Tensor(teNo,2):zero()

print('==> Testing with testing data..')


for pnum = 1,teNo  do
	testData = {
		data = test.data[pnum];
		labels = test.labels[{{pnum}}];
		size = function() return 1 end;
	}

	for t = 1, #testData.data do
		-- testData.data[t] = (testData.data[t]:mul(std)):add(mean)
		testData.data[t] = testData.data[t]:cuda()
	end

	out = mod:forward(testData.data);
	ef_True = testData.labels:float()
	efList[{pnum,{1}}] = ef_True
	out = out:float()
	efList[{pnum,{2}}] = out
	ef_err[{{pnum}}] = ef_True - out

-- 	print(out[1]*500 ..' '.. out[2]*600)
-- print(testData.labels[{1,1}]*500 ..' '.. testData.labels[{1,2}]*600)
	-- sysCalc = out[1]*500
	-- diaCalc = out[2]*600
	-- sysTrue = testData.labels[{1,1}]*500
	-- diaTrue = testData.labels[{1,2}]*600

	-- err = torch.Tensor(2)
	-- err[1] = torch.abs(sysTrue-sysCalc)
	-- err[2] = torch.abs(diaTrue-diaCalc)
	
	-- efTrue = (diaTrue-sysTrue)/diaTrue
	-- efCalc = (diaCalc-sysCalc)/diaCalc

	-- err_list[{pnum-start+1}] = err 
	-- ef_err[{pnum-start+1}] = efTrue-efCalc
	-- sum_err = sum_err+err
end


mt = require('fb.mattorch')
mt.save('ef_test.mat',efList)

-- print(torch.mean(ef_err))
-- print(efList)
end

-- gnuplot.hist(sys,100)
-- gnuplot.hist(dia,100)

-- print(sum_err)

