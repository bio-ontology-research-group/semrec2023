### This scripts wraps an ontology in Functional Syntax to be read by the OWL API

import sys

input_file = sys.argv[1]
id_ = sys.argv[2]
output_file = input_file.replace(".owl", "_wrapped.owl") if input_file.endswith(".owl") else input_file + "_wrapped.owl"

prefix = f"""
Prefix(owl:=<http://www.w3.org/2002/07/owl#>)
Prefix(rdf:=<http://www.w3.org/1999/02/22-rdf-syntax-ns#>)
Prefix(xml:=<http://www.w3.org/XML/1998/namespace>)
Prefix(xsd:=<http://www.w3.org/2001/XMLSchema#>)
Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)
Ontology(<owlapi:ontology:{id_}">
"""

content = open(input_file).read()
out_content = prefix + content + ")"
open(output_file, "w").write(out_content)
print("Done!")

