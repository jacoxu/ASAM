#!/bin/bash
#author: [jacoxu](https://github.com/jacoxu)

thchs30_folder=./data_thchs30/data/

echo "======== train data ========"
mkdir ./train
mkdir ./train/spk01_female_ZHU
cp ${thchs30_folder}"A2_1.wav" ./train/spk01_female_ZHU/A2_1.wav
cp ${thchs30_folder}"A2_3.wav" ./train/spk01_female_ZHU/A2_3.wav
cp ${thchs30_folder}"A2_5.wav" ./train/spk01_female_ZHU/A2_5.wav
cp ${thchs30_folder}"A2_7.wav" ./train/spk01_female_ZHU/A2_7.wav
cp ${thchs30_folder}"A2_9.wav" ./train/spk01_female_ZHU/A2_9.wav
cp ${thchs30_folder}"A2_11.wav" ./train/spk01_female_ZHU/A2_11.wav
cp ${thchs30_folder}"A2_13.wav" ./train/spk01_female_ZHU/A2_13.wav

mkdir ./train/spk02_female_LIU
cp ${thchs30_folder}"A4_16.wav" ./train/spk02_female_LIU/A4_16.wav
cp ${thchs30_folder}"A4_18.wav" ./train/spk02_female_LIU/A4_18.wav
cp ${thchs30_folder}"A4_20.wav" ./train/spk02_female_LIU/A4_20.wav
cp ${thchs30_folder}"A4_22.wav" ./train/spk02_female_LIU/A4_22.wav
cp ${thchs30_folder}"A4_24.wav" ./train/spk02_female_LIU/A4_24.wav
cp ${thchs30_folder}"A4_26.wav" ./train/spk02_female_LIU/A4_26.wav
cp ${thchs30_folder}"A4_28.wav" ./train/spk02_female_LIU/A4_28.wav

mkdir ./train/spk03_female_WANG
cp ${thchs30_folder}"A6_31.wav" ./train/spk03_female_WANG/A6_31.wav
cp ${thchs30_folder}"A6_33.wav" ./train/spk03_female_WANG/A6_33.wav
cp ${thchs30_folder}"A6_35.wav" ./train/spk03_female_WANG/A6_35.wav
cp ${thchs30_folder}"A6_37.wav" ./train/spk03_female_WANG/A6_37.wav
cp ${thchs30_folder}"A6_39.wav" ./train/spk03_female_WANG/A6_39.wav
cp ${thchs30_folder}"A6_41.wav" ./train/spk03_female_WANG/A6_41.wav
cp ${thchs30_folder}"A6_43.wav" ./train/spk03_female_WANG/A6_43.wav

mkdir ./train/spk04_female_GUO
cp ${thchs30_folder}"A7_46.wav" ./train/spk04_female_GUO/A7_46.wav
cp ${thchs30_folder}"A7_48.wav" ./train/spk04_female_GUO/A7_48.wav
cp ${thchs30_folder}"A7_50.wav" ./train/spk04_female_GUO/A7_50.wav
cp ${thchs30_folder}"A7_52.wav" ./train/spk04_female_GUO/A7_52.wav
cp ${thchs30_folder}"A7_54.wav" ./train/spk04_female_GUO/A7_54.wav
cp ${thchs30_folder}"A7_56.wav" ./train/spk04_female_GUO/A7_56.wav
cp ${thchs30_folder}"A7_58.wav" ./train/spk04_female_GUO/A7_58.wav

mkdir ./train/spk05_male_WANG
cp ${thchs30_folder}"A8_61.wav" ./train/spk05_male_WANG/A8_61.wav
cp ${thchs30_folder}"A8_63.wav" ./train/spk05_male_WANG/A8_63.wav
cp ${thchs30_folder}"A8_65.wav" ./train/spk05_male_WANG/A8_65.wav
cp ${thchs30_folder}"A8_67.wav" ./train/spk05_male_WANG/A8_67.wav
cp ${thchs30_folder}"A8_69.wav" ./train/spk05_male_WANG/A8_69.wav
cp ${thchs30_folder}"A8_71.wav" ./train/spk05_male_WANG/A8_71.wav
cp ${thchs30_folder}"A8_73.wav" ./train/spk05_male_WANG/A8_73.wav

mkdir ./train/spk06_male_YANG
cp ${thchs30_folder}"A33_76.wav" ./train/spk06_male_YANG/A33_76.wav
cp ${thchs30_folder}"A33_78.wav" ./train/spk06_male_YANG/A33_78.wav
cp ${thchs30_folder}"A33_80.wav" ./train/spk06_male_YANG/A33_80.wav
cp ${thchs30_folder}"A33_82.wav" ./train/spk06_male_YANG/A33_82.wav
cp ${thchs30_folder}"A33_84.wav" ./train/spk06_male_YANG/A33_84.wav
cp ${thchs30_folder}"A33_86.wav" ./train/spk06_male_YANG/A33_86.wav
cp ${thchs30_folder}"A33_88.wav" ./train/spk06_male_YANG/A33_88.wav

mkdir ./train/spk07_female_SONG
cp ${thchs30_folder}"B11_296.wav" ./train/spk07_female_SONG/B11_296.wav
cp ${thchs30_folder}"B11_298.wav" ./train/spk07_female_SONG/B11_298.wav
cp ${thchs30_folder}"B11_300.wav" ./train/spk07_female_SONG/B11_300.wav
cp ${thchs30_folder}"B11_302.wav" ./train/spk07_female_SONG/B11_302.wav
cp ${thchs30_folder}"B11_304.wav" ./train/spk07_female_SONG/B11_304.wav
cp ${thchs30_folder}"B11_306.wav" ./train/spk07_female_SONG/B11_306.wav
cp ${thchs30_folder}"B11_308.wav" ./train/spk07_female_SONG/B11_308.wav

mkdir ./train/spk08_female_ZHANG
cp ${thchs30_folder}"B12_266.wav" ./train/spk08_female_ZHANG/B12_266.wav
cp ${thchs30_folder}"B12_268.wav" ./train/spk08_female_ZHANG/B12_268.wav
cp ${thchs30_folder}"B12_270.wav" ./train/spk08_female_ZHANG/B12_270.wav
cp ${thchs30_folder}"B12_272.wav" ./train/spk08_female_ZHANG/B12_272.wav
cp ${thchs30_folder}"B12_274.wav" ./train/spk08_female_ZHANG/B12_274.wav
cp ${thchs30_folder}"B12_276.wav" ./train/spk08_female_ZHANG/B12_276.wav
cp ${thchs30_folder}"B12_278.wav" ./train/spk08_female_ZHANG/B12_278.wav

mkdir ./train/spk09_female_WANG
cp ${thchs30_folder}"B15_281.wav" ./train/spk09_female_WANG/B15_281.wav
cp ${thchs30_folder}"B15_283.wav" ./train/spk09_female_WANG/B15_283.wav
cp ${thchs30_folder}"B15_285.wav" ./train/spk09_female_WANG/B15_285.wav
cp ${thchs30_folder}"B15_287.wav" ./train/spk09_female_WANG/B15_287.wav
cp ${thchs30_folder}"B15_289.wav" ./train/spk09_female_WANG/B15_289.wav
cp ${thchs30_folder}"B15_291.wav" ./train/spk09_female_WANG/B15_291.wav
cp ${thchs30_folder}"B15_293.wav" ./train/spk09_female_WANG/B15_293.wav

mkdir ./train/spk10_female_TAN
cp ${thchs30_folder}"B31_311.wav" ./train/spk10_female_TAN/B31_311.wav
cp ${thchs30_folder}"B31_313.wav" ./train/spk10_female_TAN/B31_313.wav
cp ${thchs30_folder}"B31_315.wav" ./train/spk10_female_TAN/B31_315.wav
cp ${thchs30_folder}"B31_317.wav" ./train/spk10_female_TAN/B31_317.wav
cp ${thchs30_folder}"B31_319.wav" ./train/spk10_female_TAN/B31_319.wav
cp ${thchs30_folder}"B31_321.wav" ./train/spk10_female_TAN/B31_321.wav
cp ${thchs30_folder}"B31_323.wav" ./train/spk10_female_TAN/B31_323.wav

echo "======== dev data ========"
mkdir ./dev
mkdir ./dev/spk01_female_ZHU
cp ${thchs30_folder}"A2_2.wav" ./dev/spk01_female_ZHU/A2_2.wav
cp ${thchs30_folder}"A2_6.wav" ./dev/spk01_female_ZHU/A2_6.wav
cp ${thchs30_folder}"A2_10.wav" ./dev/spk01_female_ZHU/A2_10.wav

mkdir ./dev/spk02_female_LIU
cp ${thchs30_folder}"A4_17.wav" ./dev/spk02_female_LIU/A4_17.wav
cp ${thchs30_folder}"A4_21.wav" ./dev/spk02_female_LIU/A4_21.wav
cp ${thchs30_folder}"A4_25.wav" ./dev/spk02_female_LIU/A4_25.wav

mkdir ./dev/spk03_female_WANG
cp ${thchs30_folder}"A6_32.wav" ./dev/spk03_female_WANG/A6_32.wav
cp ${thchs30_folder}"A6_36.wav" ./dev/spk03_female_WANG/A6_36.wav
cp ${thchs30_folder}"A6_40.wav" ./dev/spk03_female_WANG/A6_40.wav

mkdir ./dev/spk04_female_GUO
cp ${thchs30_folder}"A7_47.wav" ./dev/spk04_female_GUO/A7_47.wav
cp ${thchs30_folder}"A7_51.wav" ./dev/spk04_female_GUO/A7_51.wav
cp ${thchs30_folder}"A7_55.wav" ./dev/spk04_female_GUO/A7_55.wav

mkdir ./dev/spk05_male_WANG
cp ${thchs30_folder}"A8_62.wav" ./dev/spk05_male_WANG/A8_62.wav
cp ${thchs30_folder}"A8_66.wav" ./dev/spk05_male_WANG/A8_66.wav
cp ${thchs30_folder}"A8_70.wav" ./dev/spk05_male_WANG/A8_70.wav

mkdir ./dev/spk06_male_YANG
cp ${thchs30_folder}"A33_77.wav" ./dev/spk06_male_YANG/A33_77.wav
cp ${thchs30_folder}"A33_81.wav" ./dev/spk06_male_YANG/A33_81.wav
cp ${thchs30_folder}"A33_85.wav" ./dev/spk06_male_YANG/A33_85.wav

mkdir ./dev/spk07_female_SONG
cp ${thchs30_folder}"B11_297.wav" ./dev/spk07_female_SONG/B11_297.wav
cp ${thchs30_folder}"B11_301.wav" ./dev/spk07_female_SONG/B11_301.wav
cp ${thchs30_folder}"B11_305.wav" ./dev/spk07_female_SONG/B11_305.wav

mkdir ./dev/spk08_female_ZHANG
cp ${thchs30_folder}"B12_267.wav" ./dev/spk08_female_ZHANG/B12_267.wav
cp ${thchs30_folder}"B12_271.wav" ./dev/spk08_female_ZHANG/B12_271.wav
cp ${thchs30_folder}"B12_275.wav" ./dev/spk08_female_ZHANG/B12_275.wav

mkdir ./dev/spk09_female_WANG
cp ${thchs30_folder}"B15_282.wav" ./dev/spk09_female_WANG/B15_282.wav
cp ${thchs30_folder}"B15_286.wav" ./dev/spk09_female_WANG/B15_286.wav
cp ${thchs30_folder}"B15_290.wav" ./dev/spk09_female_WANG/B15_290.wav

mkdir ./dev/spk10_female_TAN
cp ${thchs30_folder}"B31_312.wav" ./dev/spk10_female_TAN/B31_312.wav
cp ${thchs30_folder}"B31_316.wav" ./dev/spk10_female_TAN/B31_316.wav
cp ${thchs30_folder}"B31_320.wav" ./dev/spk10_female_TAN/B31_320.wav

echo "======== test data ========"
mkdir ./test
mkdir ./test/spk01_female_ZHU
cp ${thchs30_folder}"A2_0.wav" ./test/spk01_female_ZHU/A2_0.wav
cp ${thchs30_folder}"A2_4.wav" ./test/spk01_female_ZHU/A2_4.wav
cp ${thchs30_folder}"A2_8.wav" ./test/spk01_female_ZHU/A2_8.wav
cp ${thchs30_folder}"A2_12.wav" ./test/spk01_female_ZHU/A2_12.wav
cp ${thchs30_folder}"A2_14.wav" ./test/spk01_female_ZHU/A2_14.wav

mkdir ./test/spk02_female_LIU
cp ${thchs30_folder}"A4_15.wav" ./test/spk02_female_LIU/A4_15.wav
cp ${thchs30_folder}"A4_19.wav" ./test/spk02_female_LIU/A4_19.wav
cp ${thchs30_folder}"A4_23.wav" ./test/spk02_female_LIU/A4_23.wav
cp ${thchs30_folder}"A4_27.wav" ./test/spk02_female_LIU/A4_27.wav
cp ${thchs30_folder}"A4_29.wav" ./test/spk02_female_LIU/A4_29.wav

mkdir ./test/spk03_female_WANG
cp ${thchs30_folder}"A6_30.wav" ./test/spk03_female_WANG/A6_30.wav
cp ${thchs30_folder}"A6_34.wav" ./test/spk03_female_WANG/A6_34.wav
cp ${thchs30_folder}"A6_38.wav" ./test/spk03_female_WANG/A6_38.wav
cp ${thchs30_folder}"A6_42.wav" ./test/spk03_female_WANG/A6_42.wav
cp ${thchs30_folder}"A6_44.wav" ./test/spk03_female_WANG/A6_44.wav

mkdir ./test/spk04_female_GUO
cp ${thchs30_folder}"A7_45.wav" ./test/spk04_female_GUO/A7_45.wav
cp ${thchs30_folder}"A7_49.wav" ./test/spk04_female_GUO/A7_49.wav
cp ${thchs30_folder}"A7_53.wav" ./test/spk04_female_GUO/A7_53.wav
cp ${thchs30_folder}"A7_57.wav" ./test/spk04_female_GUO/A7_57.wav
cp ${thchs30_folder}"A7_59.wav" ./test/spk04_female_GUO/A7_59.wav

mkdir ./test/spk05_male_WANG
cp ${thchs30_folder}"A8_60.wav" ./test/spk05_male_WANG/A8_60.wav
cp ${thchs30_folder}"A8_64.wav" ./test/spk05_male_WANG/A8_64.wav
cp ${thchs30_folder}"A8_68.wav" ./test/spk05_male_WANG/A8_68.wav
cp ${thchs30_folder}"A8_72.wav" ./test/spk05_male_WANG/A8_72.wav
cp ${thchs30_folder}"A8_74.wav" ./test/spk05_male_WANG/A8_74.wav

mkdir ./test/spk06_male_YANG
cp ${thchs30_folder}"A33_75.wav" ./test/spk06_male_YANG/A33_75.wav
cp ${thchs30_folder}"A33_79.wav" ./test/spk06_male_YANG/A33_79.wav
cp ${thchs30_folder}"A33_83.wav" ./test/spk06_male_YANG/A33_83.wav
cp ${thchs30_folder}"A33_87.wav" ./test/spk06_male_YANG/A33_87.wav
cp ${thchs30_folder}"A33_89.wav" ./test/spk06_male_YANG/A33_89.wav

mkdir ./test/spk07_female_SONG
cp ${thchs30_folder}"B11_295.wav" ./test/spk07_female_SONG/B11_295.wav
cp ${thchs30_folder}"B11_299.wav" ./test/spk07_female_SONG/B11_299.wav
cp ${thchs30_folder}"B11_303.wav" ./test/spk07_female_SONG/B11_303.wav
cp ${thchs30_folder}"B11_307.wav" ./test/spk07_female_SONG/B11_307.wav
cp ${thchs30_folder}"B11_309.wav" ./test/spk07_female_SONG/B11_309.wav

mkdir ./test/spk08_female_ZHANG
cp ${thchs30_folder}"B12_265.wav" ./test/spk08_female_ZHANG/B12_265.wav
cp ${thchs30_folder}"B12_269.wav" ./test/spk08_female_ZHANG/B12_269.wav
cp ${thchs30_folder}"B12_273.wav" ./test/spk08_female_ZHANG/B12_273.wav
cp ${thchs30_folder}"B12_277.wav" ./test/spk08_female_ZHANG/B12_277.wav
cp ${thchs30_folder}"B12_279.wav" ./test/spk08_female_ZHANG/B12_279.wav

mkdir ./test/spk09_female_WANG
cp ${thchs30_folder}"B15_280.wav" ./test/spk09_female_WANG/B15_280.wav
cp ${thchs30_folder}"B15_284.wav" ./test/spk09_female_WANG/B15_284.wav
cp ${thchs30_folder}"B15_288.wav" ./test/spk09_female_WANG/B15_288.wav
cp ${thchs30_folder}"B15_292.wav" ./test/spk09_female_WANG/B15_292.wav
cp ${thchs30_folder}"B15_294.wav" ./test/spk09_female_WANG/B15_294.wav

mkdir ./test/spk10_female_TAN
cp ${thchs30_folder}"B31_310.wav" ./test/spk10_female_TAN/B31_310.wav
cp ${thchs30_folder}"B31_314.wav" ./test/spk10_female_TAN/B31_314.wav
cp ${thchs30_folder}"B31_318.wav" ./test/spk10_female_TAN/B31_318.wav
cp ${thchs30_folder}"B31_322.wav" ./test/spk10_female_TAN/B31_322.wav
cp ${thchs30_folder}"B31_324.wav" ./test/spk10_female_TAN/B31_324.wav

echo "======== unk data ========"
mkdir ./unk
mkdir ./unk/test
mkdir ./unk/test/spk11_male_LI
cp ${thchs30_folder}"B21_252.wav" ./test/spk11_male_LI/B21_252.wav
cp ${thchs30_folder}"B21_255.wav" ./test/spk11_male_LI/B21_255.wav
cp ${thchs30_folder}"B21_258.wav" ./test/spk11_male_LI/B21_258.wav
cp ${thchs30_folder}"B21_261.wav" ./test/spk11_male_LI/B21_261.wav
cp ${thchs30_folder}"B21_264.wav" ./test/spk11_male_LI/B21_264.wav

mkdir ./unk/test/spk12_female_ZHANG
cp ${thchs30_folder}"B32_327.wav" ./test/spk12_female_ZHANG/B32_327.wav
cp ${thchs30_folder}"B32_330.wav" ./test/spk12_female_ZHANG/B32_330.wav
cp ${thchs30_folder}"B32_333.wav" ./test/spk12_female_ZHANG/B32_333.wav
cp ${thchs30_folder}"B32_336.wav" ./test/spk12_female_ZHANG/B32_336.wav
cp ${thchs30_folder}"B32_339.wav" ./test/spk12_female_ZHANG/B32_339.wav

mkdir ./unk/test/spk13_female_YANG
cp ${thchs30_folder}"C12_502.wav" ./test/spk13_female_YANG/C12_502.wav
cp ${thchs30_folder}"C12_505.wav" ./test/spk13_female_YANG/C12_505.wav
cp ${thchs30_folder}"C12_508.wav" ./test/spk13_female_YANG/C12_508.wav
cp ${thchs30_folder}"C12_511.wav" ./test/spk13_female_YANG/C12_511.wav
cp ${thchs30_folder}"C12_514.wav" ./test/spk13_female_YANG/C12_514.wav

mkdir ./unk/test/spk14_female_LIN
cp ${thchs30_folder}"C23_517.wav" ./test/spk14_female_LIN/C23_517.wav
cp ${thchs30_folder}"C23_520.wav" ./test/spk14_female_LIN/C23_520.wav
cp ${thchs30_folder}"C23_523.wav" ./test/spk14_female_LIN/C23_523.wav
cp ${thchs30_folder}"C23_526.wav" ./test/spk14_female_LIN/C23_526.wav
cp ${thchs30_folder}"C23_529.wav" ./test/spk14_female_LIN/C23_529.wav

mkdir ./unk/test/spk15_female_LV
cp ${thchs30_folder}"D13_752.wav" ./test/spk15_female_LV/D13_752.wav
cp ${thchs30_folder}"D13_755.wav" ./test/spk15_female_LV/D13_755.wav
cp ${thchs30_folder}"D13_758.wav" ./test/spk15_female_LV/D13_758.wav
cp ${thchs30_folder}"D13_761.wav" ./test/spk15_female_LV/D13_761.wav
cp ${thchs30_folder}"D13_764.wav" ./test/spk15_female_LV/D13_764.wav

mkdir ./unk/sounds
mkdir ./unk/sounds/spk11_male_LI
cp ${thchs30_folder}"B21_250.wav" ./sounds/spk11_male_LI/B21_250.wav
cp ${thchs30_folder}"B21_251.wav" ./sounds/spk11_male_LI/B21_251.wav
cp ${thchs30_folder}"B21_253.wav" ./sounds/spk11_male_LI/B21_253.wav
cp ${thchs30_folder}"B21_254.wav" ./sounds/spk11_male_LI/B21_254.wav
cp ${thchs30_folder}"B21_256.wav" ./sounds/spk11_male_LI/B21_256.wav
cp ${thchs30_folder}"B21_257.wav" ./sounds/spk11_male_LI/B21_257.wav
cp ${thchs30_folder}"B21_259.wav" ./sounds/spk11_male_LI/B21_259.wav
cp ${thchs30_folder}"B21_260.wav" ./sounds/spk11_male_LI/B21_260.wav
cp ${thchs30_folder}"B21_262.wav" ./sounds/spk11_male_LI/B21_262.wav
cp ${thchs30_folder}"B21_263.wav" ./sounds/spk11_male_LI/B21_263.wav

mkdir ./unk/sounds/spk12_female_ZHANG
cp ${thchs30_folder}"B32_325.wav" ./sounds/spk12_female_ZHANG/B32_325.wav
cp ${thchs30_folder}"B32_326.wav" ./sounds/spk12_female_ZHANG/B32_326.wav
cp ${thchs30_folder}"B32_328.wav" ./sounds/spk12_female_ZHANG/B32_328.wav
cp ${thchs30_folder}"B32_329.wav" ./sounds/spk12_female_ZHANG/B32_329.wav
cp ${thchs30_folder}"B32_331.wav" ./sounds/spk12_female_ZHANG/B32_331.wav
cp ${thchs30_folder}"B32_332.wav" ./sounds/spk12_female_ZHANG/B32_332.wav
cp ${thchs30_folder}"B32_334.wav" ./sounds/spk12_female_ZHANG/B32_334.wav
cp ${thchs30_folder}"B32_335.wav" ./sounds/spk12_female_ZHANG/B32_335.wav
cp ${thchs30_folder}"B32_337.wav" ./sounds/spk12_female_ZHANG/B32_337.wav
cp ${thchs30_folder}"B32_338.wav" ./sounds/spk12_female_ZHANG/B32_338.wav

mkdir ./unk/sounds/spk13_female_YANG
cp ${thchs30_folder}"C12_500.wav" ./sounds/spk13_female_YANG/C12_500.wav
cp ${thchs30_folder}"C12_501.wav" ./sounds/spk13_female_YANG/C12_501.wav
cp ${thchs30_folder}"C12_503.wav" ./sounds/spk13_female_YANG/C12_503.wav
cp ${thchs30_folder}"C12_504.wav" ./sounds/spk13_female_YANG/C12_504.wav
cp ${thchs30_folder}"C12_506.wav" ./sounds/spk13_female_YANG/C12_506.wav
cp ${thchs30_folder}"C12_507.wav" ./sounds/spk13_female_YANG/C12_507.wav
cp ${thchs30_folder}"C12_509.wav" ./sounds/spk13_female_YANG/C12_509.wav
cp ${thchs30_folder}"C12_510.wav" ./sounds/spk13_female_YANG/C12_510.wav
cp ${thchs30_folder}"C12_512.wav" ./sounds/spk13_female_YANG/C12_512.wav
cp ${thchs30_folder}"C12_513.wav" ./sounds/spk13_female_YANG/C12_513.wav

mkdir ./unk/sounds/spk14_female_LIN
cp ${thchs30_folder}"C23_515.wav" ./sounds/spk14_female_LIN/C23_515.wav
cp ${thchs30_folder}"C23_516.wav" ./sounds/spk14_female_LIN/C23_516.wav
cp ${thchs30_folder}"C23_518.wav" ./sounds/spk14_female_LIN/C23_518.wav
cp ${thchs30_folder}"C23_519.wav" ./sounds/spk14_female_LIN/C23_519.wav
cp ${thchs30_folder}"C23_521.wav" ./sounds/spk14_female_LIN/C23_521.wav
cp ${thchs30_folder}"C23_522.wav" ./sounds/spk14_female_LIN/C23_522.wav
cp ${thchs30_folder}"C23_524.wav" ./sounds/spk14_female_LIN/C23_524.wav
cp ${thchs30_folder}"C23_525.wav" ./sounds/spk14_female_LIN/C23_525.wav
cp ${thchs30_folder}"C23_527.wav" ./sounds/spk14_female_LIN/C23_527.wav
cp ${thchs30_folder}"C23_528.wav" ./sounds/spk14_female_LIN/C23_528.wav

mkdir ./unk/sounds/spk15_female_LV
cp ${thchs30_folder}"D13_750.wav" ./sounds/spk15_female_LV/D13_750.wav
cp ${thchs30_folder}"D13_751.wav" ./sounds/spk15_female_LV/D13_751.wav
cp ${thchs30_folder}"D13_753.wav" ./sounds/spk15_female_LV/D13_753.wav
cp ${thchs30_folder}"D13_754.wav" ./sounds/spk15_female_LV/D13_754.wav
cp ${thchs30_folder}"D13_756.wav" ./sounds/spk15_female_LV/D13_756.wav
cp ${thchs30_folder}"D13_757.wav" ./sounds/spk15_female_LV/D13_757.wav
cp ${thchs30_folder}"D13_759.wav" ./sounds/spk15_female_LV/D13_759.wav
cp ${thchs30_folder}"D13_760.wav" ./sounds/spk15_female_LV/D13_760.wav
cp ${thchs30_folder}"D13_762.wav" ./sounds/spk15_female_LV/D13_762.wav
cp ${thchs30_folder}"D13_763.wav" ./sounds/spk15_female_LV/D13_763.wav
