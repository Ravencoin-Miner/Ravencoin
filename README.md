# Ravencoin Miner

Kraww!

An optimized fork of ccminer developed specially for x16r.

Based on Christian Buchner's &amp; Christian H.'s CUDA project, no longer active on github since 2014.

Check the [README.txt](README.txt) for the additions


# Example .bat file fail over script

This is the example .bat file provided with the miner. Feel free to use this or your own .bat file.

	:: Kraww!
	:: Set the developer donation percent with --donate. Minimum donation is 0%.
	:MINE
	ccminer -a x16r -o stratum+tcp://stratum.threeeyed.info:3333 -u RBHsbmpDrce5B7woYnRDKtMnGetz1QHGUX -p x -i 20 --donate 1 -r 5 -N 600
	ccminer -a x16r -o stratum+tcp://stratum.threeeyed.info:3333 -u RBHsbmpDrce5B7woYnRDKtMnGetz1QHGUX -p x -i 20 --donate 1 -r 3 -N 600
	ccminer -a x16r -o stratum+tcp://stratum.threeeyed.info:3333 -u RBHsbmpDrce5B7woYnRDKtMnGetz1QHGUX -p x -i 20 --donate 1 -r 3 -N 600
	ccminer -a x16r -o stratum+tcp://stratum.threeeyed.info:3333 -u RBHsbmpDrce5B7woYnRDKtMnGetz1QHGUX -p x -i 20 --donate 1 -r 3 -N 600
	GOTO :MINE

- Replace the pool connection information with your preferred Ravencoin pool
- Replace the wallet address with your own
- Most Ravencoin pools can have anything after -p, generally it is used as a worker name
- Set Intensity with -i, default is 20
- Set Donation % with --donate
- Set number of times miner will try to reconnect to pool before moving to next connection in script with -r
- -N 600 Makes the mining program use 600 shares in the calculation of the average hash rate that is displayed on Accepted share lines
- This has the effect of stabilizing the displayed hash rate



## Donation Addresses

Consider supporting the contributors to this miner by donating to the following addresses:

--Banshee (developer of Ravencoin miner)

- RVN: RBHsbmpDrce5B7woYnRDKtMnGetz1QHGUX

Built from source on Windows 10 x64


Compile on Linux
----------------

Please see [INSTALL](https://github.com/tpruvot/ccminer/blob/linux/INSTALL) file or [project Wiki](https://github.com/tpruvot/ccminer/wiki/Compatibility)
