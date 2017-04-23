require 'torch';   -- torch
require 'image' ;  -- for image transforms
require 'nn' ;     -- provides all sorts of trainable modules/layers
require 'cudnn';
require 'rnn';
require 'cunn';
py = require('fb.python')

total = 696
trNo = 600
teNo = total-trNo
ex = 700 - total

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
	-- R and S
sysVolumes = py.eval('sysVolumes')
diaVolumes = py.eval('diaVolumes')
ef = py.eval('ef')

-- inputSize = 110000
inputSize = 6875
outputSize = 1
rho = 30

print('==> Finding mean and std..')

mainFold = '/home/brats/athira/LSTMdata625'
data_fold = '../LSTMdata625/pnum_'
--finding mean and std

-- k = 1
-- local d = torch.Tensor(trNo*rho,inputSize):zero()
-- d = d:cuda()	
-- for pnum = 1,trNo+ex do
-- 	if pnum~=416 and pnum~=499 and pnum~=597 and pnum~=33 then

-- 		temp = torch.load(data_fold..pnum..'.t7')
-- 		for i = 1,rho do
-- 			d[{k}] = temp[i]
-- 			k = k+1
-- 		end
-- 	end
-- end

-- mean = d:mean()
-- std = d:std()
-- d = d:t()
-- torch.save('dataMean.t7',mean)
-- torch.save('dataStd.t7',std)

--load mean and std?

mean = torch.load('dataMean.t7')
std = torch.load('dataStd.t7')

--Training data
data = {}
labels = torch.Tensor(1)
print('==> Loading training data..')

k=1
for pnum = 1,trNo+ex do
	if pnum~=416 and pnum~=499 and pnum~=597 and pnum~= 33  then
		temp = torch.load(data_fold..pnum..'.t7')
		n = #temp
		if k ==1 then
			labels[{1}] = ef[pnum]
		else
			temp2 = torch.Tensor(outputSize)
			temp2[{1}] = ef[pnum]
			labels = torch.cat(labels,temp2,1)
		end
		for i = 1,n do
			temp[i] = temp[i]:add(-mean)
			temp[i] = temp[i]:div(std)
		end
		table.insert(data,temp)
		k = k+1
		-- labels[{k,2}] = sysVolumes[pnum]
		-- labels[{k,3}] = diaVolumes[pnum]

		-- Augment imbalanced data
		if (ef[pnum]>=0.5 and ef[pnum]<0.55) or (ef[pnum]>=0.65 and ef[pnum]<0.75) then
			-- augment once: hflips
			temp = torch.load(mainFold..'/hflip625/pnum_'..pnum..'.t7')
			n = #temp
			temp2 = torch.Tensor(outputSize)
			temp2[{1}] = ef[pnum]
			labels = torch.cat(labels,temp2,1)
			for i = 1,n do
				temp[i] = temp[i]:add(-mean)
				temp[i] = temp[i]:div(std)
			end
			table.insert(data,temp)
			k = k+1

			else if ef[pnum]<0.5 or ef[pnum]>=0.75 then
				--augment thrice
				temp = torch.load(mainFold..'/hflip625/pnum_'..pnum..'.t7')
				n = #temp
				temp2 = torch.Tensor(outputSize)
			temp2[{1}] = ef[pnum]
			labels = torch.cat(labels,temp2,1)
				for i = 1,n do
					temp[i] = temp[i]:add(-mean)
					temp[i] = temp[i]:div(std)
				end
				table.insert(data,temp)
				k = k+1

				temp = torch.load(mainFold..'/vflip625/pnum_'..pnum..'.t7')
				n = #temp
				temp2 = torch.Tensor(outputSize)
			temp2[{1}] = ef[pnum]
			labels = torch.cat(labels,temp2,1)
				for i = 1,n do
					temp[i] = temp[i]:add(-mean)
					temp[i] = temp[i]:div(std)
				end
				table.insert(data,temp)
				k = k+1

				temp = torch.load(mainFold..'/transpose625/pnum_'..pnum..'.t7')
				n = #temp
				temp2 = torch.Tensor(outputSize)
			temp2[{1}] = ef[pnum]
			labels = torch.cat(labels,temp2,1)
				for i = 1,n do
					temp[i] = temp[i]:add(-mean)
					temp[i] = temp[i]:div(std)
				end
				table.insert(data,temp)
				k = k+1

			end
		end

		if ef[pnum]<0.4 or ef[pnum]>=0.75 then
				--augment thrice again
				temp = torch.load(mainFold..'/hflip625/pnum_'..pnum..'.t7')
				n = #temp
				temp2 = torch.Tensor(outputSize)
			temp2[{1}] = ef[pnum]
			labels = torch.cat(labels,temp2,1)
				for i = 1,n do
					temp[i] = temp[i]:add(-mean)
					temp[i] = temp[i]:div(std)
				end
				table.insert(data,temp)
				k = k+1

				temp = torch.load(mainFold..'/vflip625/pnum_'..pnum..'.t7')
				n = #temp
				temp2 = torch.Tensor(outputSize)
			temp2[{1}] = ef[pnum]
			labels = torch.cat(labels,temp2,1)
				for i = 1,n do
					temp[i] = temp[i]:add(-mean)
					temp[i] = temp[i]:div(std)
				end
				table.insert(data,temp)
				k = k+1

				temp = torch.load(mainFold..'/transpose625/pnum_'..pnum..'.t7')
				n = #temp
				temp2 = torch.Tensor(outputSize)
			temp2[{1}] = ef[pnum]
			labels = torch.cat(labels,temp2,1)
				for i = 1,n do
					temp[i] = temp[i]:add(-mean)
					temp[i] = temp[i]:div(std)
				end
				table.insert(data,temp)
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

trNo = k

data = {}
labels = torch.Tensor(1)
print('==> Loading testing data..')

k=1
for pnum = 700-teNo+1,700 do
	if pnum~=416 and pnum~=499 and pnum~=597 and pnum~= 33  then
		temp = torch.load(data_fold..pnum..'.t7')
		n = #temp
		if k ==1 then
			labels[{1}] = ef[pnum]
		else
			temp2 = torch.Tensor(outputSize)
			temp2[{1}] = ef[pnum]
			labels = torch.cat(labels,temp2,1)
		end
		for i = 1,n do
			temp[i] = temp[i]:add(-mean)
			temp[i] = temp[i]:div(std)
		end
		table.insert(data,temp)
		k = k+1
	end
end

testData = {
	data = data;
	labels = labels;
}

print('==> Saving testing data..')
torch.save('testData.t7',testData)

teNo = k

print(trNo)
print(teNo)


