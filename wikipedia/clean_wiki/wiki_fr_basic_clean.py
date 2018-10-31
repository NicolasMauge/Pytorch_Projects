# Basic cleaning of the wikipedia text (text balise in wikipedia dump)
# Copyright 2018, Nicolas MAUGE
import re 

quiet=True

def process_line(text):
    ignore=False
    for index, line in enumerate(text.split("\n")):
        line=line.strip() 
        if '{{infobox' in line.lower() \
            or "wikitable" in line \
            or "<gallery" in line or "<blockquote" in line \
            or line=="{{Autres projets" or "{{colonnes" in line:   # specific for french
            ignore = True # ignore the lines until 'ignore' becomes False again
            if not quiet:   print(f"Ignore a group of lines {line}({index})")
        elif ignore and (line =="}}" or line=="" or line=="|}" or line=="</gallery>" \
            or line=="</blockquote>"):
            ignore = False
            if not quiet:   print(f"Last line ignored {line} ({index})")
        elif not ignore and line!="" and "DEFAULTSORT" not in line \
            and line[:12]!="[[Catégorie:": # specific for french
            yield line

def clean_commas(line):
    return line.replace("'''", "").replace("''", "")

def process_finditer(regex, line, function_match):
    previous_end = None
    for m in regex.finditer(line):
        start = m.start()
        end = m.end()
        match = m.group()
        if previous_end:
            line_processed = line_processed+ line[previous_end:start]+function_match(match)
            previous_end = end
        else:
            line_processed = line[:start] + function_match(match)
            previous_end = end
    if previous_end:
        line_processed = line_processed + line[previous_end:]
    else:
        line_processed = line
    return line_processed

def clean_b_simple(line):
    re_curly_only = re.compile(r'\{\{[^\{^\|^\}]+\}\}')
    re_brackets_only = re.compile(r'\[\[[^\[^\|^\]]+\]\]')
    
    for regex in [re_curly_only, re_brackets_only]:
        line = process_finditer(regex, line, lambda match:match[2:-2])

    return line

def clean_b_simple_pipe(line):
    re_simple_pipe = re.compile(r'\[\[[^\[^\]](.*?)\|(.*?)\]\]')
    re_simple_pipe2 = re.compile(r'\{\{[^\{^\}](.*?)\|(.*?)\}\}')

    def function_match(match):
        m_split = match.split("|")
        if len(m_split)==2:
            if "citation" in m_split[0].lower():
                return '« '+m_split[1][:-2]+' »'
            else:
                return m_split[1][:-2]
        elif len(m_split)>=3:
            if "citation" in m_split[0].lower():
                return '« '+m_split[1] + " (" + m_split[2][:-2]+') »'
            else:
                return m_split[1] + " (" + m_split[2][:-2]+")"
        else:
            return match

    for regex in [re_simple_pipe, re_simple_pipe2]:
        line = process_finditer(regex, line, function_match)
    
    return line

def clean_comments(line):
    re_comments = re.compile(r'<\!(.+?)>')
    return re_comments.sub("", line)

def clean_date(line):
    re_date = re.compile(r'\{\{(date|Date)[^\}](.+?)\}\}')

    def function_match(match):
        match_split = match.split("|")
        if len(match_split)==4:
            return match_split[1]+" "+ match_split[2] + " " + match_split[3][:-2]
        else:
            return match

    line = process_finditer(re_date, line, function_match)
    
    return line

def clean_unite(line):
    re_unite = re.compile(r'\{\{unité(.+?)[^\}]\}\}')

    def function_match(match):
        match_split = match.split("|")
        if len(match_split)==3:
            return match_split[1]+" "+ match_split[2][:-2]
        else:
            return match

    line = process_finditer(re_unite, line, function_match)
    
    return line


def clean_ref(line): 
    line = re.compile(r'<(r|R)ef>(.*?)[^>]/>').sub("", line)
    line = re.compile(r'<(r|R)ef>(.*?)[^>]</ref>').sub("", line)
    line = re.compile(r'<(r|R)ef(.*?)[^>]>(.*?)[^>]</ref>').sub("", line)
    line = re.compile(r'<ref name=(.*?)[^>]>(.*?)[^>]/ref>').sub("", line)
    line = re.compile(r'<ref name=(.*?)[^>]/>').sub("", line)

    return line

def clean_equals(line):
    equals = ["=====", "====", "===", "=="]
    for equal in equals:
        if line[:len(equal)]==equal:
            line = "#"*len(equal)+line[len(equal):].replace(equal, "")

    return line

def suppr_lines(line):
    begins = ["[[fichier", "{{légende", "{{article détaillé", "{{ébauche", "{{pertinence", \
            "{{arbre", "|", "[[image", "{{autres projets", "[[file", "! ", "</blockquote>", \
            "{{palette", "{{portail", "contenu=", "* [http:", "thumb", "bar:"]
    for begin in begins:
        if line[:len(begin)].lower() == begin:
            return ""
    return line 

def last_clean(line):
    return line.replace("<center>", "").replace("</center>", "").replace("<br>", "").replace("<br />", "").strip()

def spec_ml_select_lines(line):
    """
        select a line if it is useful for a machine learning algorithm, ie 
            - not less than 9 words (like a short title)
    """
    if len(line.split(" "))>=10:
        return line
    else:
        return ""

def suppr_return(text):
    text = re.compile(r"<ref>(.*?)[^>]</ref>", re.MULTILINE|re.DOTALL).sub("", text)
    
    return text

def clean_text(text):
    cleaned = []

    text = suppr_return(text)

    for index, line in enumerate(process_line(text)):
        # basic cleaning
        process_proc = [clean_date, clean_ref, clean_comments, clean_b_simple, 
                        clean_b_simple_pipe, clean_unite, clean_commas, clean_equals, \
                        suppr_lines, last_clean]
        for proc in process_proc:
            line = proc(line)

        # specific to machine learning
        line = spec_ml_select_lines(line)

        if line == "### Bibliographie" or line=="## Bibliographie":
            break
        elif line!="":
            cleaned.append(line)

        last_index = index

    cleaned = ' x_return '.join(cleaned) # one line for one article, 'x_return' for \n

    return last_index, cleaned
