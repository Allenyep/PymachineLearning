class Codec:

    url_table = {}
    count = 0
    
    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.
        
        :type longUrl: str
        :rtype: str
        """
        self.count = self.count + 1
        self.url_table[longUrl] = 'http://'+str(self.count)
        return self.url_table[longUrl]
        
        

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.
        
        :type shortUrl: str
        :rtype: str
        """
        # list(dicxx.keys())[list(dicxx.values()).index("001")]
        return list(self.url_table.keys())[list(self.url_table.values()).index(shortUrl)]
    
    def get_keys(self, d, value):
        return [k for k,v in d.items() if v == value]

# Your Codec object will be instantiated and called as such:
codec = Codec()
# codec.decode(codec.encode(url))

# print(codec.encode("http://www.leetcode.com/faq/?id=10"))
print(codec.decode(codec.encode("http://www.leetcode.com/faq/?id=10")))