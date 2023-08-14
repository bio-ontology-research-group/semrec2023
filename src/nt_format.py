# fix the format of a nt file

import sys

input_filename = sys.argv[1]
if not input_filename.endswith('.nt'):
    raise Exception('input file must be .nt')


with open(input_filename, 'r') as f:
    text = f.read()
    corrected_text = text.replace('.<', '.\n<')

output_filename = input_filename.replace('.nt', '_corrected.nt')
with open(output_filename, 'w') as f:
    f.write(corrected_text)


print("Done!")
