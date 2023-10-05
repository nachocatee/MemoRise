package com.tjjhtjh.memorise.domain.memo.service;

import com.tjjhtjh.memorise.domain.item.exception.NoItemException;
import com.tjjhtjh.memorise.domain.item.repository.ItemRepository;
import com.tjjhtjh.memorise.domain.item.repository.entity.Item;
import com.tjjhtjh.memorise.domain.memo.exception.BookmarkException;
import com.tjjhtjh.memorise.domain.memo.exception.MemoException;
import com.tjjhtjh.memorise.domain.memo.repository.BookmarkRepository;
import com.tjjhtjh.memorise.domain.memo.repository.MemoRepository;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Bookmark;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.memo.service.dto.request.BookmarkRequest;
import com.tjjhtjh.memorise.domain.memo.service.dto.request.MemoRequest;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoCountResponse;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoDetailResponse;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoResponse;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MyMemoResponse;
import com.tjjhtjh.memorise.domain.tag.repository.TaggedTeamRepository;
import com.tjjhtjh.memorise.domain.tag.repository.TaggedUserRepository;
import com.tjjhtjh.memorise.domain.tag.service.dto.request.TaggedTeamRequest;
import com.tjjhtjh.memorise.domain.tag.service.dto.request.TaggedUserRequest;
import com.tjjhtjh.memorise.domain.team.exception.NoTeamException;
import com.tjjhtjh.memorise.domain.team.repository.TeamRepository;
import com.tjjhtjh.memorise.domain.team.repository.entity.Team;
import com.tjjhtjh.memorise.domain.user.exception.NoUserException;
import com.tjjhtjh.memorise.domain.user.repository.UserRepository;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;

@Slf4j
@RequiredArgsConstructor
@Transactional(readOnly = true)
@Service
public class MemoService {

    private final MemoRepository memoRepository;
    private final UserRepository userRepository;
    private final BookmarkRepository bookMarkRepository;
    private final ItemRepository itemRepository;
    private final TaggedUserRepository taggedUserRepository;
    private final TaggedTeamRepository taggedTeamRepository;
    private final TeamRepository teamRepository;

    private static final String NO_USER_EMAIL = "이메일에 해당하는 유저가 없습니다";
    private static final String NO_USER = "해당하는 유저가 존재하지 않습니다";
    private static final String NO_MEMO = "해당하는 메모가 없습니다";
    private static final String NO_FIND_BOOKMARK = "해당하는 북마크를 찾을 수 없습니다";
    private static final String NO_FIND_ITEM = "해당하는 아이템을 찾을 수 없습니다";
    private static final String NO_TEAM_EXCEPTION = "해당하는 팀을 찾을 수 없습니다";

    @Transactional
    public void createMemo(MemoRequest memoRequest, String itemName) throws MemoException {
        User user = userRepository.findByUserSeqAndIsDeletedFalse(memoRequest.getUserId())
                .orElseThrow(() -> new NoUserException(NO_USER_EMAIL));
        Item item = itemRepository.findByItemName(itemName)
                .orElseThrow(() -> new NoItemException(NO_FIND_ITEM));
        // 파일 있을 때 없을 때 저장 로직
        if (memoRequest.getNewFile() == null) {
            memoRepository.save(memoRequest.registToEntity(user, item));
        } else if (memoRequest.getNewFile().isBlank()) {
            memoRepository.save(memoRequest.registToEntity(user, item));
        } else {
            memoRepository.save(memoRequest.registToEntity(user, item, memoRequest.getNewFile()));
        }

        Memo memo = memoRepository.findByLastSaveData(user.getUserSeq()).orElseThrow(() -> new MemoException(NO_MEMO));

        firstMemoAddUserTag(memoRequest.getTaggedUserList(), memo);
        firstMemoAddTeamTag(memoRequest.getTaggedTeamList(), memo);

    }

    @Transactional
    public void firstMemoAddUserTag(List<Long> userList, Memo memo) {
        if (userList.size() > 0) {
            TaggedUserRequest taggedUserRequest = new TaggedUserRequest();
            for (Long userSeq : userList) {
                User user = userRepository.findByUserSeqAndIsDeletedFalse(userSeq)
                        .orElseThrow(() -> new NoUserException(NO_USER));
                taggedUserRepository.save(taggedUserRequest.saveUserToEntity(memo, user));
            }
        }
    }

    @Transactional
    public void firstMemoAddTeamTag(List<Long> teamList, Memo memo) {
        if (teamList.size() > 0) {
            TaggedTeamRequest taggedTeamRequest = new TaggedTeamRequest();
            for (Long teamSeq : teamList) {
                Team team = teamRepository.findByTeamSeqAndIsDeletedFalse(teamSeq)
                        .orElseThrow(() -> new NoTeamException(NO_TEAM_EXCEPTION));
                taggedTeamRepository.save(taggedTeamRequest.saveTeamToEntity(memo, team));
            }
        }
    }

    @Transactional
    public void updateMemo(MemoRequest memoRequest, Long memoId, String itemName) throws MemoException {
        Memo memo = memoRepository.findById(memoId)
                .orElseThrow(() -> new MemoException(NO_MEMO));
        User user = userRepository.findByUserSeqAndIsDeletedFalse(memo.getUser().getUserSeq())
                .orElseThrow(() -> new NoUserException(NO_USER_EMAIL));
        Item item = itemRepository.findByItemName(itemName)
                .orElseThrow(() -> new NoItemException(NO_FIND_ITEM));

        if (memo.getFile() == null && memoRequest.getNewFile() == null) {
            memoRepository.save(memoRequest.updateToNullFileEntity(memoId, memoRequest, user, item));
        } else if (memo.getFile() == null && memoRequest.getNewFile() != null) {
            memoRepository.save(memoRequest.updateToEntity(memoId, memoRequest, user, item));
        } else if (memo.getFile() != null && memoRequest.getNewFile() == null) {
            memoRepository.save(memoRequest.updateToNoChangeFileEntity(memoId, memoRequest, user, item, memoRequest.getNewFile()));
        } else if (memo.getFile() != null && memoRequest.getNewFile() != null) {
            memoRepository.save(memoRequest.updateToEntity(memoId, memoRequest, user, item));
        }

    }

    @Transactional
    public void fakeDeleteMemo(Long memoId, MemoRequest memoRequest) throws MemoException {
        Memo memo = memoRepository.findById(memoId)
                .orElseThrow(() -> new MemoException(NO_MEMO));
        User user = userRepository.findByUserSeqAndIsDeletedFalse(memo.getUser().getUserSeq())
                .orElseThrow(() -> new NoUserException(NO_USER_EMAIL));
        Item item = itemRepository.findByItemName(memoRequest.getItemName()).orElseThrow(() -> new NoItemException(NO_FIND_ITEM));

        memoRepository.save(memoRequest.deleteToEntity(memo, user, item));
        List<Bookmark> bookmarkList = bookMarkRepository.fakeDeleteMemoBookmarkList(memoId);

        if (bookmarkList.size()>0) {
            for (Bookmark bookmark : bookmarkList) {
                bookMarkRepository.delete(bookmark);
            }
        }
    }

    @Transactional
    public void addBookmark(Long memoId, Long userSeq) throws MemoException {
        Memo memo = memoRepository.findById(memoId).orElseThrow(() -> new MemoException(NO_MEMO));
        User user = userRepository.findByUserSeqAndIsDeletedFalse(userSeq)
                .orElseThrow(() -> new UsernameNotFoundException(NO_USER));

        bookMarkRepository.save(BookmarkRequest.saveToEntity(memo, user));
    }

    @Transactional
    public void deleteBookmark(Long memoId, Long userSeq) {
        User user = userRepository.findByUserSeqAndIsDeletedFalse(userSeq)
                .orElseThrow(() -> new NoUserException(NO_USER));
        Bookmark bookmark = bookMarkRepository.findByMemoAndUser(memoId,userSeq)
                .orElseThrow(() -> new BookmarkException(NO_FIND_BOOKMARK));
        bookMarkRepository.delete(bookmark);
    }

    public List<MemoResponse> itemMemoView(String itemName, Long userSeq) {
        Item item = itemRepository.findByItemName(itemName).orElseThrow(() -> new NoItemException(NO_FIND_ITEM));
        User user = userRepository.findByUserSeqAndIsDeletedFalse(userSeq).orElseThrow(() -> new NoUserException(NO_USER));

        return memoRepository.findWrittenByMeOrOpenMemoOrTaggedMemo(item.getItemSeq(), userSeq);
    }

    public MemoDetailResponse detailMemo(Long memoId, Long userSeq) throws MemoException {
        User user = userRepository.findByUserSeqAndIsDeletedFalse(userSeq).orElseThrow(() -> new NoUserException(NO_USER));

        MemoDetailResponse memoDetailResponse = memoRepository.detailMemo(memoId, userSeq)
                .orElseThrow(() -> new MemoException(NO_MEMO));

        return new MemoDetailResponse().detailResponse(
                memoDetailResponse,
                bookMarkRepository.bookmarkBoolean(memoId, userSeq),
                taggedUserRepository.findByTaggedUserList(memoId),
                taggedTeamRepository.findByTaggedTeamList(memoId));
    }

    public List<MyMemoResponse> myMemoList(Long userSeq) {
        User user = userRepository.findByUserSeqAndIsDeletedFalse(userSeq).orElseThrow(() -> new NoUserException(NO_USER));
        return memoRepository.findByMyMemoIsDeletedFalse(userSeq);
    }

    public List<MyMemoResponse> allMyMemoList(Long userSeq) {
        User user = userRepository.findByUserSeqAndIsDeletedFalse(userSeq).orElseThrow(() -> new NoUserException(NO_USER));
        return memoRepository.findByAllMyMemoIsDeletedFalse(userSeq);
    }

    public List<MemoCountResponse> countOfMemoList(Long userSeq) {
        User user = userRepository.findByUserSeqAndIsDeletedFalse(userSeq).orElseThrow(() -> new NoUserException(NO_USER));
        List<MemoCountResponse> resultList = new ArrayList<>();
        List<Long> itemSeqList = itemRepository.itemSeqList();
        for (Long itemSeq : itemSeqList) {
            Item item = itemRepository.findByItemSeq(itemSeq).orElseThrow(() -> new NoItemException(NO_FIND_ITEM));
            resultList.add(new MemoCountResponse().countResponse(item.getItemName(), memoRepository.countMemoOfItem(itemSeq, userSeq)));
        }
        return resultList;
    }

    public List<MyMemoResponse> allBookmarkTrueList(Long userSeq) {
        User user = userRepository.findByUserSeqAndIsDeletedFalse(userSeq).orElseThrow(() -> new NoUserException(NO_USER));
        return bookMarkRepository.isBookmarkTrueList(userSeq);
    }
}
