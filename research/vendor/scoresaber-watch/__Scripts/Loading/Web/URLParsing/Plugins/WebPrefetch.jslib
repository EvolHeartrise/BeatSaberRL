mergeInto(LibraryManager.library, {

  $ArcViewerPrefetch: {
    getMessages: function() {
      if (!window.__arcViewerPrefetchMessages) {
        window.__arcViewerPrefetchMessages = {
          nextId: 1,
          items: {}
        };
      }

      return window.__arcViewerPrefetchMessages;
    },

    storeBytes: function(bytes) {
      var messages = ArcViewerPrefetch.getMessages();
      var messageId = messages.nextId++;
      messages.items[messageId] = bytes;
      return messageId;
    },

    toBytes: function(data) {
      if (!data) return null;
      if (data instanceof ArrayBuffer) return new Uint8Array(data);
      if (ArrayBuffer.isView && ArrayBuffer.isView(data)) {
        return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      }

      return null;
    },

    sendResult: function(requestId, gameObjectName, messageId) {
      SendMessage(gameObjectName, "OnArcViewerPrefetch", JSON.stringify({
        requestId: requestId,
        messageId: messageId || 0
      }));
    }
  },

  ArcViewerTakePrefetched__deps: ["$ArcViewerPrefetch"],
  ArcViewerPrefetchedLength__deps: ["$ArcViewerPrefetch"],
  ArcViewerCopyPrefetched__deps: ["$ArcViewerPrefetch"],

  ArcViewerTakePrefetched: function(requestId, url, gameObjectName) {
    url = UTF8ToString(url);
    gameObjectName = UTF8ToString(gameObjectName);

    var prefetches = window.__arcPrefetch;
    if (!prefetches || !Object.prototype.hasOwnProperty.call(prefetches, url)) {
      return 0;
    }

    var promise = prefetches[url];
    delete prefetches[url];

    Promise.resolve(promise)
      .then(function(data) {
        var bytes = ArcViewerPrefetch.toBytes(data);
        var messageId = bytes ? ArcViewerPrefetch.storeBytes(bytes) : 0;
        ArcViewerPrefetch.sendResult(requestId, gameObjectName, messageId);
      })
      .catch(function() {
        ArcViewerPrefetch.sendResult(requestId, gameObjectName, 0);
      });

    return 1;
  },

  ArcViewerPrefetchedLength: function(messageId) {
    var bytes = ArcViewerPrefetch.getMessages().items[messageId];
    return bytes ? bytes.length : -1;
  },

  ArcViewerCopyPrefetched: function(messageId, target, length) {
    var messages = ArcViewerPrefetch.getMessages();
    var bytes = messages.items[messageId];
    if (!bytes || length < bytes.length) {
      return 0;
    }

    HEAPU8.set(bytes, target);
    delete messages.items[messageId];
    return bytes.length;
  }
});
