// 라이브러리
import axios from "axios";
import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  Image,
  Pressable,
} from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { useSelector } from "react-redux";

// 스타일
import { calculateDynamicWidth } from "../../../constants/dynamicSize";
import Colors from "../../../constants/colors";

// 백엔드 통신
import { BACKEND_URL } from "../../../util/http";

// 컴포넌트
import BookMarkBtn from "../../Button/BookMarkBtn";
import { RootState } from "../../../store/store";

// 메인페이지 상태관리를 위한 타입 지정
type MemoListProp = {
  onMemoWritePress: () => void;
  onMemoDetailPress: (memoSeq: number) => void;
  id: string | null;
  memoStatus: string;
  memoListStatus: boolean;
  onMemoZeroLength: (checker: boolean) => void;
};

const MemoList: React.FC<MemoListProp> = ({
  onMemoWritePress,
  onMemoDetailPress,
  id,
  memoStatus,
  memoListStatus,
  onMemoZeroLength,
}) => {
  // 유저ID
  const userId = useSelector((state: RootState) => state.userInfo.id);

  // FlatList 사용을 위한 Type 지정
  type MemoTypeProps = {
    accessType: string;
    content: string;
    file: string | null;
    isBookmarked: boolean;
    itemImage: string;
    memoSeq: number;
    nickname: string;
    updatedAt: string;
  };

  type MemoListItemProps = {
    item: MemoTypeProps;
  };

  //날짜 변경 함수
  const formatData = (inputData: string) => {
    const date = new Date(inputData);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");

    return `${year}. ${month}. ${day}`;
  };

  // 북마크 상태관리 및 함수
  const [memoData, setMemoData] = useState<MemoTypeProps[]>([]);

  // 메모 리스트 AXIOS (main이면 MainScreen / 아니면 MenuMemo)
  useEffect(() => {
    const fetchData = async () => {
      try {
        let response;
        if (memoStatus === "main") {
          response = await axios.get(
            BACKEND_URL + `/memos/${id}/list/${userId}`
          );
          if (response.data.length === 0) {
            onMemoZeroLength(true);
          }
        } else {
          response = await axios.get(
            BACKEND_URL +
              `/user/${userId}/${
                memoStatus === "saved"
                  ? "bookmarks"
                  : memoStatus === "all"
                  ? "memos"
                  : "my-memos"
              }`
          );
        }
        setMemoData(response.data);
      } catch (error) {
        console.log(error);
      }
    };
    fetchData();
  }, [memoListStatus]);

  // FlatList 사용을 위한 MemoList 정리
  const MemoList: React.FC<MemoListItemProps> = ({ item }) => (
    <View style={styles.memoContainer}>
      <Pressable onPress={() => onMemoDetailPress(item.memoSeq)}>
        <LinearGradient
          colors={["#FFFFFF", "#F5F5F5"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 0, y: 1 }}
          style={{ minHeight: calculateDynamicWidth(104) }}
        >
          <View style={styles.innerContainer}>
            <View style={styles.calenderContainer}>
              <Text style={styles.calendar}>{formatData(item.updatedAt)}</Text>
            </View>
            {item.file !== null ? (
              <View>
                <Text
                  style={styles.content}
                  numberOfLines={2}
                  ellipsizeMode="tail"
                >
                  {item.content}
                </Text>
                {item.file && (
                  <Image source={{ uri: item.file }} style={styles.photo} />
                )}
              </View>
            ) : (
              <Text
                style={styles.content}
                numberOfLines={4}
                ellipsizeMode="tail"
              >
                {item.content}
              </Text>
            )}
            <View style={styles.bottomContainer}>
              <Text style={styles.nickname}>{item.nickname}</Text>
              <Text style={styles.open}>
                {item.accessType === "OPEN"
                  ? "전체공개"
                  : item.accessType === "RESTRICT"
                  ? "일부공개"
                  : "비공개"}
              </Text>
            </View>
          </View>
        </LinearGradient>
      </Pressable>
      <BookMarkBtn
        memoSeq={item.memoSeq}
        detailStyle={[styles.bookmark, styles.bookmarkSize]}
        bookmarkType={item.isBookmarked}
      />
    </View>
  );

  return (
    <>
      {/* memoStatus에 따라서 메모 작성 버튼 유무 판단 */}
      {memoStatus === "main" ? (
        <View style={styles.mainContainer_main}>
          {memoData.length !== 0 && (
            <Pressable
              style={styles.memoWriteContainer}
              onPress={onMemoWritePress}
            >
              <Image
                source={require("../../../assets/icons/memo_write.png")}
                style={styles.memoWrite}
              />
            </Pressable>
          )}
          <View style={styles.memoListContainer_main}>
            <FlatList
              data={memoData}
              renderItem={({ item }) => <MemoList item={item} />}
              keyExtractor={(item) => item.memoSeq.toString()}
            />
          </View>
        </View>
      ) : (
        <View style={styles.mainContainer_menu}>
          <View style={styles.memoListContainer_menu}>
            <FlatList
              data={memoData}
              renderItem={({ item }) => <MemoList item={item} />}
              keyExtractor={(item) => item.memoSeq.toString()}
            />
          </View>
        </View>
      )}
    </>
  );
};

export default MemoList;

const styles = StyleSheet.create({
  mainContainer_main: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [
      { translateY: -calculateDynamicWidth(350) / 2 },
      { translateX: -calculateDynamicWidth(306) / 2 },
    ],
  },

  mainContainer_menu: {
    position: "absolute",
    top: "43%",
    left: "50%",
    transform: [
      { translateY: -calculateDynamicWidth(350) / 2 },
      { translateX: -calculateDynamicWidth(306) / 2 },
    ],
  },

  memoWriteContainer: {
    alignItems: "flex-end",
    marginBottom: calculateDynamicWidth(3),
  },

  memoWrite: {
    width: calculateDynamicWidth(38),
    height: calculateDynamicWidth(38),
  },

  memoListContainer_main: {
    height: calculateDynamicWidth(350),
  },

  memoListContainer_menu: {
    height: calculateDynamicWidth(570),
  },

  memoContainer: {
    width: calculateDynamicWidth(306),
    minHeight: calculateDynamicWidth(104),
    maxHeight: calculateDynamicWidth(185),
    borderRadius: calculateDynamicWidth(15),
    marginBottom: calculateDynamicWidth(12),
    overflow: "scroll",
  },

  innerContainer: {
    margin: calculateDynamicWidth(9),
    flex: 1,
    justifyContent: "space-between",
  },

  bookmark: {
    position: "absolute",
    top: 0,
    marginLeft: calculateDynamicWidth(16),
  },

  bookmarkSize: {
    width: calculateDynamicWidth(21),
    height: calculateDynamicWidth(28.41),
  },

  calenderContainer: {
    alignItems: "flex-end",
  },

  calendar: {
    color: Colors.hover,
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(14),
  },

  content: {
    color: Colors.text,
    fontSize: calculateDynamicWidth(18),
    fontFamily: "Pretendard-Regular",
    marginVertical: calculateDynamicWidth(5),
    marginHorizontal: calculateDynamicWidth(10),
  },

  photo: {
    width: calculateDynamicWidth(275),
    height: calculateDynamicWidth(60),
    borderRadius: calculateDynamicWidth(10),
    marginLeft: calculateDynamicWidth(5),
  },

  bottomContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
  },

  nickname: {
    color: Colors.blue500,
    fontFamily: "Pretendard-Medium",
    fontSize: calculateDynamicWidth(14),
  },

  open: {
    color: "rgba(76, 106, 255, 0.6)",
    fontFamily: "Pretendard-Medium",
    fontSize: calculateDynamicWidth(14),
  },
});
