mergeInto(LibraryManager.library, {

  GetParameters: function() {
    var str = window.location.search;
    var bufferSize = lengthBytesUTF8(str) + 1;
    var buffer = _malloc(bufferSize);
    stringToUTF8(str, buffer, bufferSize);
    return buffer;
  },

  SetPageTitle: function(title) {
    title = UTF8ToString(title);
    document.title = title;
  },

  GetArcViewerEnv: function(name) {
    name = UTF8ToString(name);

    var env = window.arcviewerEnv || window.__ARCVIEWER_ENV__ || {};
    var aliases = {
      ARCVIEWER_BASE_URL: "arcViewerBaseUrl",
      ARCVIEWER_SCORESABER_BASE_URL: "scoreSaberBaseUrl",
      ARCVIEWER_SCORESABER_API_URL: "scoreSaberApiUrl"
    };

    var value = env[name] || env[aliases[name]];
    if (!value) {
      return 0;
    }

    var str = String(value);
    var bufferSize = lengthBytesUTF8(str) + 1;
    var buffer = _malloc(bufferSize);
    stringToUTF8(str, buffer, bufferSize);
    return buffer;
  },

  OpenExternalURL: function(url) {
    url = UTF8ToString(url);

    var opened = null;
    try {
      opened = window.open(url, "_blank", "noopener,noreferrer");
    } catch (err) {
      opened = null;
    }

    if (opened) {
      return;
    }

    try {
      window.top.open(url, "_blank", "noopener,noreferrer");
      return;
    } catch (err) {}

    try {
      var link = document.createElement("a");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      return;
    } catch (err) {}

    try {
      window.top.location.href = url;
    } catch (err) {
      window.location.href = url;
    }
  }
});