# adapted from:
# Simple example of streaming a Wikipedia 
# Copyright 2017 by Jeff Heaton, released under the The GNU Lesser General Public License (LGPL).
# http://www.heatonresearch.com
import os

def process_xml_dump():
	import xml.etree.ElementTree as etree
	
	FILENAME_WIKI = 'frwiki-latest-pages-articles.xml'
	PATH_WIKI_XML = 'data/wiki_dumps/'
	pathWikiXML = os.path.join(PATH_WIKI_XML, FILENAME_WIKI)

	totalCount = 0
	articleCount = 0
	title = None
	article_number = 0

	def strip_tag_name(t):
	    t = elem.tag
	    idx = k = t.rfind("}")
	    if idx != -1:
	        t = t[idx + 1:]
	    return t

	for event, elem in etree.iterparse(pathWikiXML, events=('start', 'end')):
		tname = strip_tag_name(elem.tag)

		if event == 'start':
			if tname == 'page':
				title = ''
				id = -1
				redirect = ''
				inrevision = False
				ns = 0
			elif tname == 'revision':
                # Do not pick up on revision id's
				inrevision = True
		else:
			if tname == 'title':
				title = elem.text
			elif tname == 'id' and not inrevision:
				id = int(elem.text)
			elif tname == 'redirect':
				redirect = elem.attrib['title']
			elif tname == 'ns':
				ns = int(elem.text)
			elif tname == 'text':
				text_balise = elem.text
			elif tname == 'page':
				totalCount += 1
            
				if(ns !=10):
					if((text_balise is not None)&(text_balise !="")):
						if(text_balise.find("#REDIRECT") == -1)&(text_balise.find("#redirection") == -1):
							article_number = article_number + 1
							text_balise = "Article : " + title + "\n" + text_balise
							yield text_balise

			elem.clear()