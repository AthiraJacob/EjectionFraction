require 'torch';   -- torch
require 'image' ;  -- for image transforms
require 'nn' ;     -- provides all sorts of trainable modules/layers
require 'cudnn';
require 'rnn';
require 'cunn';

total = 696
trNo = 600
teNo = total-trNo
augment = 0
ex = 700 - total

py = require('fb.python')


-- Load volumes
py.exec([=[
import numpy as np
from openpyxl import load_workbook
wb = load_workbook(filename = '../data.xlsx',data_only = 1)
sh = wb['Sheet']
sysVolumes = np.zeros(700)
diaVolumes = np.zeros(700)
ef = np.zeros(700)
for i in range(700):
	sysVolumes[i] = sh['R'+str(i+2)].value
	diaVolumes[i] = sh['S'+str(i+2)].value
	ef[i] = sh['M'+str(i+2)].value
	]=])

sysVolumes = py.eval('sysVolumes')
diaVolumes = py.eval('diaVolumes')
ef = py.eval('ef')

print('==> Finding mean and std..')

--finding mean and std

inputSize = 55000
outputSize = 3

k = 1
if augment ==1 then
	 d = torch.Tensor(2*trNo*30,inputSize):zero()	
else
	 d = torch.Tensor(trNo*30,inputSize):zero()
end

-- d = d:cuda()	
for pnum = 1,trNo+ex do
	if pnum~=416 and pnum~=499 and pnum~=597 and pnum~=33 then

		temp = torch.load('../LSTMdata/pnum_'..pnum..'.t7')
		for i = 1,30 do
			d[{k}] = temp[i][{{1,inputSize}}]
			k = k+1
		end
	end
end

if augment ==1 then
	print('---- Using augmented data ----')
	for pnum = 1,trNo+ex do
	if pnum~=416 and pnum~=499 and pnum~=597 and pnum~=33 then

		temp = torch.load('../LSTMdata/data_5deg_cl/pnum_'..pnum..'.t7')
		for i = 1,30 do
			d[{k}] = temp[i][{{1,inputSize}}]
			k = k+1
		end
	end
end
end

mean = d:mean()
std = d:std()
d = d:t()
torch.save('dataMean.t7',mean)
torch.save('dataStd.t7',std)

-- --load mean and std?

mean = torch.load('dataMean.t7')
std = torch.load('dataStd.t7')



--Training data
data = {}
if augment == 1 then
	labels = torch.Tensor(2*trNo,outputSize)
else
	labels = torch.Tensor(trNo,outputSize)
end

print('==> Loading training data..')

k=1
for pnum = 1,trNo+ex do
	if pnum~=416 and pnum~=499 and pnum~=597 and pnum~= 33  then
		temp = torch.load('../LSTMdata/pnum_'..pnum..'.t7')
		n = #temp
		for i = 1,n do
			temp[i] = temp[i][{{1,inputSize}}]:add(-mean)
			temp[i] = temp[i][{{1,inputSize}}]:div(std)
		end
		table.insert(data,temp)
		labels[{k,1}] = sysVolumes[pnum]
		labels[{k,2}] = diaVolumes[pnum]
		labels[{k,3}] = ef[pnum]
		k = k+1
	end
end

if augment == 1 then
	for pnum = 1,trNo+ex do
	if pnum~=416 and pnum~=499 and pnum~=597 and pnum~= 33  then
		temp = torch.load('../LSTMdata/data_5deg_cl/pnum_'..pnum..'.t7')
		n = #temp
		for i = 1,n do
			temp[i] = temp[i][{{1,inputSize}}]:add(-mean)
			temp[i] = temp[i][{{1,inputSize}}]:div(std)
		end
		table.insert(data,temp)
		labels[{k,1}] = sysVolumes[pnum]
		labels[{k,2}] = diaVolumes[pnum]
		labels[{k,3}] = ef[pnum]
		k = k+1
	end
end
end

trainData = {
	data = data;
	labels = labels;
}

print('==> Saving training data..')
torch.save('trainData.t7',trainData)



--Testing data
data = {}
labels = torch.Tensor(teNo,outputSize)
print('==> Loading testing data..')

k = 1
for pnum = trNo+ex+1,700 do
	if pnum~=416 and pnum~=499 and pnum~=597 and pnum~= 33 then

		temp = torch.load('../LSTMdata/pnum_'..pnum..'.t7')
		n = #temp
		for i = 1,n do
			temp[i] = temp[i][{{1,inputSize}}]:add(-mean)
			temp[i] = temp[i][{{1,inputSize}}]:div(std)
		end
		table.insert(data,temp)
		-- labels[{k,1}] = sysVolumes[pnum]
		-- labels[{k,2}] = diaVolumes[pnum]
		labels[{k,1}] = ef[pnum]
		k = k+1
	end	
end

testData = {
	data = data;
	labels = labels;
}

print('==> Saving testing data..')
torch.save('testData.t7',testData)

