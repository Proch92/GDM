import shareddict

prof = shareddict.SharedDict()

prof.add('a', 5)
prof.add('a', 6)
prof.add('a', 7)

print(prof.a)
