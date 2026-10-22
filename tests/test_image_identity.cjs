const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../idcard_label/static/app.js"), "utf8");

function setup() {
  const images = [];
  const context = vm.createContext({
    state: { current: { id: 1, image_md5: "old", version: 1 }, imageMd5: "old" },
    elements: { modelRefreshButton: { disabled: false, textContent: "Refresh" } },
    Image: function () { images.push(this); },
    resizeCanvas() {},
    pushUndo() {},
    markDirty() {},
    renderTargets() {},
    draw() {},
    showToast() {},
    encodeURIComponent,
  });
  for (const [start, end] of [
    ["function loadImage(", "function resizeCanvas("],
    ["async function refreshWithModel(", "function renderModelOptions("],
  ]) {
    vm.runInContext(source.slice(source.indexOf(start), source.indexOf(end)), context);
  }
  return { context, images };
}

test("reused IDs produce different image URLs and delayed old images are ignored", async () => {
  const { context, images } = setup();
  const oldLoad = context.loadImage(context.state.current);
  context.state.current = { id: 1, image_md5: "new", version: 1 };
  const newLoad = context.loadImage(context.state.current);
  assert.notEqual(images[0].src, images[1].src);
  images[1].onload();
  await newLoad;
  images[0].onload();
  await oldLoad;
  assert.equal(context.state.image, images[1]);
  assert.equal(context.state.imageMd5, "new");
});

test("model results for an old image cannot update a reused item ID", async () => {
  const { context } = setup();
  let finish;
  context.request = (url, options) => {
    assert.equal(JSON.parse(options.body).image_md5, "old");
    return new Promise((resolve) => { finish = resolve; });
  };
  const pending = context.refreshWithModel();
  context.state.current = { id: 1, image_md5: "new", annotations: [] };
  context.state.imageMd5 = "new";
  finish({ image_md5: "old", annotations: [{ class_id: 1 }] });
  await pending;
  assert.equal(context.state.current.annotations.length, 0);
});

test("model refresh requires the displayed image and response to match", async () => {
  const { context } = setup();
  let calls = 0;
  context.request = async () => {
    calls++;
    return { image_md5: "wrong", annotations: [{ class_id: 1 }] };
  };
  context.state.imageMd5 = "different";
  await context.refreshWithModel();
  assert.equal(calls, 0);
  context.state.imageMd5 = "old";
  await context.refreshWithModel();
  assert.equal(calls, 1);
  assert.equal(context.state.current.annotations, undefined);
  context.request = async () => ({ image_md5: "old", annotations: [{ class_id: 1 }] });
  await context.refreshWithModel();
  assert.equal(context.state.current.annotations[0].class_id, 1);
});
