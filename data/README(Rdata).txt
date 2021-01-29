Projections from 10k posterior samples of the Hector-BRICK model.

Full parameter chains published in https://doi.org/10.5281/zenodo.3236413 in file: TTEGICGISAIS.csv

Files:
*_quantiles.tsv:
	-Contains 0.025, 0.5, 0.975 quantiles for global annual mean projections out to 2150 under RCP2.6 and RCP8.5
	-Tgav - temperature [K]
	-slr - sea-level rise [m] 
	-gic - contribution from glaciers and small ice caps [m]
	-gis - contribution from Greenland ice sheet [m]
	-te  - contribution from thermal expansion [m]
	-ais - contribution from Antarctic ice sheet [m]
	-ocheat - ocean heat [10^22 J]
 
highECS_quantiles.tsv - projections for ECS >= 5 K
lowECS_quantiles.tsv - projections for ECS < 5 K

sample_projections.RData:
	-Includes projections and hindcasts from each member of the 10k posterior sample
	-parameters.sample - the parameter sets for each projection/hindcast
	-parnames - the names of the the parameters (the first is climate sensitivity)

*Note:
BRICK makes the modeling assumption that full glacial melt never occurs, 
	but in some high-forcing and long-term scenarios this assumption is violated, leading to model failure. 
Filtering out these incomplete simulations leaves 9996 (10,000) RCP8.5 (RCP2.6) projections to 2100,
	and 9917 (9995) RCP8.5 (RCP2.6) projections to 2150. 
