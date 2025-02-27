describe('th-has-data-cells', function () {
	'use strict';

	var fixture = document.getElementById('fixture');
	var checkContext = {
		_relatedNodes: [],
		_data: null,
		data: function (d) {
			this._data = d;
		},
		relatedNodes: function (rn) {
			this._relatedNodes = rn;
		}
	};

	afterEach(function () {
		fixture.innerHTML = '';
		checkContext._relatedNodes = [];
		checkContext._data = null;
	});

	it('should return true each row header has a non-empty cell', function (){
		fixture.innerHTML =
			'<table>' +
			'  <tr> <th>hi</th> <td>hello</td> </tr>' +
			'  <tr> <th>hi</th> <td>hello</td> </tr>' +
			'  <tr> <td>hello</td> <th>hi</th> </tr>' +
			'  <tr> <td>hello</td> <th>hi</th> </tr>' +
			'</table>';

		var node = fixture.querySelector('table');
		assert.isTrue(checks['th-has-data-cells'].evaluate.call(checkContext, node));
	});

	it('should return true each non-empty column header has a cell', function (){
		fixture.innerHTML =
			'<table>' +
			'  <tr> <th>H</th> <th>H</th> </tr>' +
			'  <tr> <td>hi</td> <td>hello</td></tr>' +
			'</table>';

		var node = fixture.querySelector('table');
		assert.isTrue(checks['th-has-data-cells'].evaluate.call(checkContext, node));
	});

	it('should return true if referred to with headers attr', function (){
		fixture.innerHTML =
			'<table>' +
			'  <tr> <td headers="a">hi</td> <td headers="b">hello</td></tr>' +
			'  <tr> <th id="a">H</th> <th id="b">H</th> </tr>' +
			'</table>';

		var node = fixture.querySelector('table');
		assert.isTrue(checks['th-has-data-cells'].evaluate.call(checkContext, node));
	});

	it('should return true if referred to with aria-labelledby', function (){
		fixture.innerHTML =
			'<table>' +
			'  <tr> <td aria-labelledby="a">hi</td> <td aria-labelledby="b">hello</td></tr>' +
			'  <tr> <th id="a">H</th> <th id="b">H</th> </tr>' +
			'</table>';

		var node = fixture.querySelector('table');
		assert.isTrue(checks['th-has-data-cells'].evaluate.call(checkContext, node));
	});

	it('should return true if the th element is empty', function (){
		fixture.innerHTML =
			'<table>' +
			'  <tr> <th></th> <th></th> </tr>' +
			'  <tr> <th></th> <th></th> </tr>' +
			'</table>';
		var node = fixture.querySelector('table');
		assert.isTrue(checks['th-has-data-cells'].evaluate.call(checkContext, node));
	});

	it('should return false if a th has no data cells', function (){
		fixture.innerHTML =
			'<table>' +
			'  <tr> <th>hi</th> </tr>' +
			'  <tr> <th>hi</th> </tr>' +
			'</table>';

		var node = fixture.querySelector('table');
		assert.isFalse(checks['th-has-data-cells'].evaluate.call(checkContext, node));
	});

	it('should return false if all data cells are empty', function (){
		fixture.innerHTML =
			'<table>' +
			'  <tr> <th>hi</th> <td></td> </tr>' +
			'  <tr> <th>hi</th> <td></td> </tr>' +
			'</table>';

		var node = fixture.querySelector('table');
		assert.isFalse(checks['th-has-data-cells'].evaluate.call(checkContext, node));
	});

	it('should return false if a td with role=columnheader is used that has no data cells', function (){
		fixture.innerHTML =
			'<table id="fail4">' +
			'  <tr> <td>aXe</td> <td role="columnheader">AXE</th> </tr>' +
			'</table>';

		var node = fixture.querySelector('table');
		assert.isFalse(checks['th-has-data-cells'].evaluate.call(checkContext, node));
	});
});