# convert nt file to owl

import rdflib
import sys

filename = sys.argv[1]
if not filename.endswith('.nt'):
    print('Not an nt file')
    sys.exit(0)

g = rdflib.Graph()
g.parse(filename, format='nt')

outfilename = filename.replace('.nt', '.owl')
with open(outfilename, 'wb') as f:
    g.serialize(destination=f, format='xml')
print('Done')

