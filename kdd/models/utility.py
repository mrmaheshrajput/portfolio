import numpy as np


# red_feat_index refers to redundant features index, the features
# that were removed because percentage of missing values was
# greater than the threshold
red_feat_index = np.array([0,1,2,3,4,7,8,9,10,11,13,14,15,16,17,18,19,22,25,26,28,29,30,31,32,33,35,36,38,39,40,
                  41,42,44,45,46,47,48,49,50,51,52,53,54,55,57,58,59,60,61,62,63,65,66,67,68,69,70,71,74,
                  76,78,79,81,83,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,
                  107,109,110,113,114,115,116,117,119,120,121,123,126,127,128,129,130,134,135,136,137,138,
                  140,141,144,145,146,147,149,150,151,153,154,155,156,157,158,160,161,163,164,165,166,167,
                  168,169,170,171,173,174,175,176,177,178,179,181,182,183,184,185,186,187,188,189,190,193,
                  199,200,208,212,213,214,223,224,228,229])


# col_names refers to column names, the exact order of columns
# after preprocessing, as per in jupyter notebook
col_names = np.array(['Var6', 'Var7', 'Var13', 'Var21', 'Var22', 'Var24', 'Var25',
       'Var28', 'Var35', 'Var38', 'Var44', 'Var57', 'Var65', 'Var73',
       'Var74', 'Var76', 'Var78', 'Var81', 'Var83', 'Var85', 'Var109',
       'Var112', 'Var113', 'Var119', 'Var123', 'Var125', 'Var126',
       'Var132', 'Var133', 'Var134', 'Var140', 'Var143', 'Var144',
       'Var149', 'Var153', 'Var160', 'Var163', 'Var173', 'Var181',
                      'Var211_freq', 'Var208_freq', 'Var218_freq',
       'Var196_freq', 'Var205_freq', 'Var223_freq', 'Var203_freq',
       'Var210_freq', 'Var221_freq', 'Var227_freq', 'Var207_freq',
       'Var206_freq', 'Var195_freq', 'Var219_freq', 'Var226_freq',
       'Var228_freq', 'Var193_freq', 'Var212_freq', 'Var204_freq',
       'Var197_freq', 'Var192_freq', 'Var216_freq', 'Var198_freq',
       'Var220_freq', 'Var222_freq', 'Var199_freq', 'Var202_freq',
       'Var217_freq',
       'Var1_indicator', 'Var2_indicator', 'Var3_indicator',
       'Var4_indicator', 'Var5_indicator', 'Var6_indicator',
       'Var7_indicator', 'Var8_indicator', 'Var9_indicator',
       'Var10_indicator', 'Var11_indicator', 'Var12_indicator',
       'Var13_indicator', 'Var14_indicator', 'Var15_indicator',
       'Var16_indicator', 'Var17_indicator', 'Var18_indicator',
       'Var19_indicator', 'Var20_indicator', 'Var21_indicator',
       'Var22_indicator', 'Var23_indicator', 'Var24_indicator',
       'Var25_indicator', 'Var26_indicator', 'Var27_indicator',
       'Var28_indicator', 'Var29_indicator', 'Var30_indicator',
       'Var31_indicator', 'Var32_indicator', 'Var33_indicator',
       'Var34_indicator', 'Var35_indicator', 'Var36_indicator',
       'Var37_indicator', 'Var38_indicator', 'Var39_indicator',
       'Var40_indicator', 'Var41_indicator', 'Var42_indicator',
       'Var43_indicator', 'Var44_indicator', 'Var45_indicator',
       'Var46_indicator', 'Var47_indicator', 'Var48_indicator',
       'Var49_indicator', 'Var50_indicator', 'Var51_indicator',
       'Var52_indicator', 'Var53_indicator', 'Var54_indicator',
       'Var55_indicator', 'Var56_indicator', 'Var57_indicator',
       'Var58_indicator', 'Var59_indicator', 'Var60_indicator',
       'Var61_indicator', 'Var62_indicator', 'Var63_indicator',
       'Var64_indicator', 'Var65_indicator', 'Var66_indicator',
       'Var67_indicator', 'Var68_indicator', 'Var69_indicator',
       'Var70_indicator', 'Var71_indicator', 'Var72_indicator',
       'Var73_indicator', 'Var74_indicator', 'Var75_indicator',
       'Var76_indicator', 'Var77_indicator', 'Var78_indicator',
       'Var79_indicator', 'Var80_indicator', 'Var81_indicator',
       'Var82_indicator', 'Var83_indicator', 'Var84_indicator',
       'Var85_indicator', 'Var86_indicator', 'Var87_indicator',
       'Var88_indicator', 'Var89_indicator', 'Var90_indicator',
       'Var91_indicator', 'Var92_indicator', 'Var93_indicator',
       'Var94_indicator', 'Var95_indicator', 'Var96_indicator',
       'Var97_indicator', 'Var98_indicator', 'Var99_indicator',
       'Var100_indicator', 'Var101_indicator', 'Var102_indicator',
       'Var103_indicator', 'Var104_indicator', 'Var105_indicator',
       'Var106_indicator', 'Var107_indicator', 'Var108_indicator',
       'Var109_indicator', 'Var110_indicator', 'Var111_indicator',
       'Var112_indicator', 'Var113_indicator', 'Var114_indicator',
       'Var115_indicator', 'Var116_indicator', 'Var117_indicator',
       'Var118_indicator', 'Var119_indicator', 'Var120_indicator',
       'Var121_indicator', 'Var122_indicator', 'Var123_indicator',
       'Var124_indicator', 'Var125_indicator', 'Var126_indicator',
       'Var127_indicator', 'Var128_indicator', 'Var129_indicator',
       'Var130_indicator', 'Var131_indicator', 'Var132_indicator',
       'Var133_indicator', 'Var134_indicator', 'Var135_indicator',
       'Var136_indicator', 'Var137_indicator', 'Var138_indicator',
       'Var139_indicator', 'Var140_indicator', 'Var141_indicator',
       'Var142_indicator', 'Var143_indicator', 'Var144_indicator',
       'Var145_indicator', 'Var146_indicator', 'Var147_indicator',
       'Var148_indicator', 'Var149_indicator', 'Var150_indicator',
       'Var151_indicator', 'Var152_indicator', 'Var153_indicator',
       'Var154_indicator', 'Var155_indicator', 'Var156_indicator',
       'Var157_indicator', 'Var158_indicator', 'Var159_indicator',
       'Var160_indicator', 'Var161_indicator', 'Var162_indicator',
       'Var163_indicator', 'Var164_indicator', 'Var165_indicator',
       'Var166_indicator', 'Var167_indicator', 'Var168_indicator',
       'Var169_indicator', 'Var170_indicator', 'Var171_indicator',
       'Var172_indicator', 'Var173_indicator', 'Var174_indicator',
       'Var175_indicator', 'Var176_indicator', 'Var177_indicator',
       'Var178_indicator', 'Var179_indicator', 'Var180_indicator',
       'Var181_indicator', 'Var182_indicator', 'Var183_indicator',
       'Var184_indicator', 'Var185_indicator', 'Var186_indicator',
       'Var187_indicator', 'Var188_indicator', 'Var189_indicator',
       'Var190_indicator', 'Var191_indicator', 'Var192_indicator',
       'Var193_indicator', 'Var194_indicator', 'Var195_indicator',
       'Var196_indicator', 'Var197_indicator', 'Var198_indicator',
       'Var199_indicator', 'Var200_indicator', 'Var201_indicator',
       'Var202_indicator', 'Var203_indicator', 'Var204_indicator',
       'Var205_indicator', 'Var206_indicator', 'Var207_indicator',
       'Var208_indicator', 'Var209_indicator', 'Var210_indicator',
       'Var211_indicator', 'Var212_indicator', 'Var213_indicator',
       'Var214_indicator', 'Var215_indicator', 'Var216_indicator',
       'Var217_indicator', 'Var218_indicator', 'Var219_indicator',
       'Var220_indicator', 'Var221_indicator', 'Var222_indicator',
       'Var223_indicator', 'Var224_indicator', 'Var225_indicator',
       'Var226_indicator', 'Var227_indicator', 'Var228_indicator',
       'Var229_indicator', 'Var230_indicator', 'num_na_count',
       'num_zero_count'])
