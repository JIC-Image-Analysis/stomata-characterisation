def exploratory_locate_stomata():
    """Locate stomata in the image."""

    def shape_factor(region):
        S = region.area
        L = region.perimeter

        return (4 * math.pi * S) / (L * L)

    composite = np.zeros(connected_components.shape)

    def separate_regions():
        for ccID in np.unique(connected_components):
            r = Region.from_id_array(connected_components, ccID)
            dr = r.dilate(10)
            h = r.convex_hull
            pre_ratio = float(r.area) / r.perimeter
            after_ratio = float(dr.area) / dr.perimeter

            #print ccID, pre_ratio, after_ratio / pre_ratio, float(h.area) / h.perimeter
            print ccID, h.area, h.perimeter, shape_factor(h)
            scipy.misc.imsave('dr{}.png'.format(ccID), dr.border.bitmap_array)
            scipy.misc.imsave('h{}.png'.format(ccID), r.convex_hull.border.bitmap_array)

            composite[h.coord_list] = ccID

        scipy.misc.imsave('composite.png', composite)

    def fit_measure(ccID):
        r = Region.from_id_array(connected_components, ccID)
        h = r.convex_hull
        em = skimage.measure.EllipseModel()
        data = zip(*h.border.coord_list)
        em.estimate(np.array(data))
        
        residuals = em.residuals(np.array(data))

        return np.inner(residuals, residuals) / (len(residuals) ** 2)


    stomata_ids = []

    def print_all_residuals():
        for ccID in np.unique(connected_components):
            r = Region.from_id_array(connected_components, ccID)
            if 100 < r.convex_hull.perimeter < 400:
                fm = fit_measure(ccID)
                print ccID, r.convex_hull.perimeter, fm
                if fm < 0.01:
                    stomata_ids.append(ccID)

    print_all_residuals()

    xdim, ydim = connected_components.shape
    annotated = np.zeros((xdim, ydim, 3))
    for ccID in stomata_ids:
        r = Region.from_id_array(connected_components, ccID)
        h = r.convex_hull
        em = skimage.measure.EllipseModel()
        data = zip(*h.border.coord_list)
        em.estimate(np.array(data))
        t_range = np.linspace(0, 2 * math.pi, 500)
        predicted = em.predict_xy(t_range).astype(np.uint16)
        annotated[h.border.coord_list] = 255, 255, 255
        annotated[zip(*predicted)] = 255, 0, 0
    scipy.misc.imsave('annotated.png', annotated)
