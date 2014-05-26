Xray-TIE
========

Computes Projected Thickness via Phase-Contrast Imaging Algorithm that uses the Transport of Intensity Equation (Paganin Phase Retrieval from Single Defocused Image).

Example application call: (This will be fleshed out more once I have all bugs fixed)
xray_tie.exe input_folder_path outout_folder_path filename_prefix startNum endNum IinVal Mag defocus_distance mu delta pixel_size regularization_param

(Unit in mm where units are appropriate)

Example:
xray_tie.exe [Path to this folder]\input [Path to this folder]\output xray_tie_input_ 0 14 1.0 1.0 30 0.00828 0.0001 .00325 0.1
