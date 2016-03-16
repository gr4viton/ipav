import subprocess
import threading
import StringIO

class terminal(threading.Thread):
    def run(self):
        self.prompt()

    def prompt(self):
        x = True
        while x:
            command = raw_input(':')
            x = self.interpret(command)

    def interpret(self,command):
        if command == 'exit':
            return False
        else:
            print 'Invalid Command'
        return True

class test(threading.Thread):
    command = 'java -jar ../bukkit/craftbukkit.jar'
    test = StringIO.StringIO()
    p = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while (p.poll() == None):
        line = p.stderr.readline()
        if not line: break
        print line.strip()

term = terminal()
testcl = test()
term.start()
testcl.start()