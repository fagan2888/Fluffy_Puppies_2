function para_se = get_log_se_non()
[A,B,C,D,E,F,G] = textread('tvmat.txt','%f %f %f %f %f %f %f');
[para, para_se] = prop_haz('log',C,B,D,[E/10000,G]);