function para = get_log_para()
[A,B,C,D,E,F,G] = textread('stmat-1.txt','%f %f %f %f %f %f %f');
[para, para_se] = prop_haz('log',C,B,D,[E/10000,G]);

