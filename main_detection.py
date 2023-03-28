def number_plate(img_file,api_json,kernel,number_model_file,text_model_file):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']=api_json
    client=vision_v1.ImageAnnotatorClient()
    floder_path=os.path.join(img_file)
    with io.open(floder_path,'rb') as image_file:
        content=image_file.read()
    image=vision_v1.types.Image(content=content)    
    response=client.object_localization(image=image)
    localized_object=response.localized_object_annotations
    objects = response.localized_object_annotations
    coord=[]
    for obj in objects:
        name = obj.name
        score = obj.score
        vertices = obj.bounding_poly.normalized_vertices
        coordinates = [(v.x, v.y) for v in vertices]
        if name=="License plate":
            coord.append(coordinates)
            print(f"{name} - {score}: {coordinates}")
    pillow_image=Image.open(floder_path)
    width, height = pillow_image.size
    left = width * coord[0][0][0]
    top = height * coord[0][1][1]
    right= width * coord[0][2][0]
    bottom = height * coord[0][2][1]
    draw = ImageDraw.Draw(pillow_image)
    area=(left,top,right,bottom)
    # draw.rectangle((left, top, right, bottom), outline="red")
    c=pillow_image.crop(area)
    c_img=np.array(c)
    a=cv2.erode(c_img,(1,1),2)
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 122, 1, cv2.THRESH_BINARY)[1]
    a1=cv2.erode(binary,(4,4),6)
    thresh = cv2.threshold(binary,0,255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]
    label_image1=measure.label(thresh,connectivity=1)
    fig,ax=plt.subplots(figsize=(10,10))
    ax.imshow(thresh,cmap='gray')
    char=[]
    column_list=[]
    lis=[]
    for i in measure.regionprops(label_image1):
        x1,y1,x2,y2=i.bbox
        roi=thresh[x1:x2,y1:y2]
        h=y2-y1
        w=x2-x1
        area=h*w
        rectangle=mpatches.Rectangle((y1,x1),h,w,linewidth=2,fill=False,edgecolor='blue')
        ax.add_patch(rectangle)
    #print(area)
        if label_image1.shape[1]>150 and label_image1.shape[0]>80:
            if area>300 and h<w:
                lis.append((i,y1))
        #print(area)
                resized_char=cv2.resize(roi,(20,20))
                char.append(resized_char)
                column_list.append(y1)
        else:
            if area>40 and h<w:
                lis.append((i,y1))
                resized_char=cv2.resize(roi,(20,20))
                char.append(resized_char)
                column_list.append(y1)
    plt.tight_layout()
    plt.show()
    print(len(lis))
    lis = sorted(lis, key=lambda x: x[1])
    sorted_i = [x[0] for x in lis]
    split=[]
    plt.figure(figsize=(10,10))
    for i in sorted_i:
        min_row, min_col, max_row, max_col = i.bbox
        cropped_image = thresh[min_row-2:max_row+2,min_col-1:max_col+1]
    #gry=cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        #plt.imshow(cropped_image, cmap='gray')
        split.append(cropped_image)
        plt.axis('off')
    print(len(sorted_i))
    loaded_model_text = tf.keras.models.load_model(text_model_file)
    loaded_model_num=tf.keras.models.load_model(number_model_file)
    letters=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    if len(lis)==10:
        char=[split[0],split[1],split[4],split[5]]
        text=[]
        for i in char:
            u=i
            uu=cv2.bitwise_not(u)
            uuu=cv2.dilate(uu,kernel,10)
            v=cv2.resize(uu,(60,60))
            aaa=v.reshape((1,60,60,1))
            color_image = np.repeat(aaa, 3, axis=-1)
            pred=loaded_model_text.predict(color_image)
            i=pred.argmax()
            c=letters[i]
            text.append(c)
            #print(text)
        letter=[split[2],split[3],split[6],split[7],split[8],split[9]]
        lett=[]
        for i in letter:
        u=i
        u=cv2.bitwise_not(u)
        uuu=cv2.dilate(uu,kernel,10)
        v=cv2.resize(u,(60,60))
        aaa=v.reshape((1,60,60,1))
        color_image = np.repeat(aaa, 3, axis=-1)
        pred=loaded_model_num.predict(color_image)
        i=pred.argmax()
        lett.append(i)
        number_plate=[text[0],text[1],lett[0],lett[1],text[2],text[3],lett[2],lett[3],lett[4],lett[5]]
        str_plate=[str(ele) for ele in number_plate]
        str_plate
        result=''.join(str_plate)
        print(result)
    elif len(lis)==9:
        char=[split[0],split[1],split[4]]
        text=[]
        for i in char:
            u=i
            uu=cv2.bitwise_not(u)
            uuu=cv2.dilate(uu,kernel,10)
            v=cv2.resize(uu,(60,60))
            aaa=v.reshape((1,60,60,1))
            color_image = np.repeat(aaa, 3, axis=-1)
            pred=loaded_model_text.predict(color_image)
            i=pred.argmax()
            c=letters[i]
            text.append(c)
            print(text)
        letter=[split[2],split[3],split[5],split[6],split[7],split[8]]
        lett=[]
        for i in letter:
            u=i
            u=cv2.bitwise_not(u)
            uuu=cv2.dilate(uu,kernel,10)
            v=cv2.resize(u,(60,60))
            aaa=v.reshape((1,60,60,1))
            color_image = np.repeat(aaa, 3, axis=-1)
            pred=loaded_model_num.predict(color_image)
            i=pred.argmax()
            lett.append(i)
        number_plate=[text[0],text[1],lett[0],lett[1],text[2],lett[2],lett[3],lett[4],lett[5]]
        str_plate=[str(ele) for ele in number_plate]
        str_plate
        result=''.join(str_plate)
        print(result)
    
