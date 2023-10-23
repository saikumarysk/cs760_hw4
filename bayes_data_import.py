import os

def read_file_from_name(filename):
    full_filename = os.path.join(os.getcwd(), 'languageID', filename)
    if not os.path.isfile(full_filename): return None
    data = []
    f_obj =  open(full_filename, 'r')
    data = list(f_obj.read().replace('\n', ''))
    f_obj.close()
    
    freq = [0]*27
    for c in data:
        if c == ' ': freq[-1] += 1
        else: freq[ord(c)-ord('a')] += 1
    
    return data, filename[0], freq

if __name__ == '__main__':
    print(read_file_from_name('e11.txt'))