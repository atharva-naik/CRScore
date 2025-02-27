import {CREDENTIALS} from "../config.func";

module.exports = function(server) {

    test('who am I?', () => {
      return server.whoami().then(function (username) {
        expect(username).toMatch(CREDENTIALS.user);
      });
    });

};

