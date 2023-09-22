package com.tjjhtjh.memorise.domain.memo.service;

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
import com.tjjhtjh.memorise.domain.user.exception.NoUserException;
import com.tjjhtjh.memorise.domain.user.repository.UserRepository;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

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
    private static final String NO_USER_EMAIL = "이메일에 해당하는 유저가 없습니다";
    private static final String NO_USER = "해당하는 유저가 존재하지 않습니다";
    private static final String NO_MEMO = "해당하는 메모가 없습니다";
    private static final String NO_FIND_BOOKMARK = "해당하는 북마크를 찾을 수 없습니다";
    private static final String NO_FIND_ITEM = "해당하는 아이템을 찾을 수 없습니다";

    @Transactional
    public void createMemo(MemoRequest memoRequest, Long itemSeq) {
        User user = userRepository.findByUserSeqAndIsDeletedFalse(memoRequest.getUserId())
                .orElseThrow(() -> new NoUserException(NO_USER_EMAIL));
        // TODO : itemException 생성 후 exception 변경 예정
        Item item = itemRepository.findByItemSeq(itemSeq)
                .orElseThrow(() -> new NullPointerException(NO_FIND_ITEM));
        // 파일 있을 때 없을 때 저장 로직
        if(memoRequest.getNewFile() == null) {
            memoRepository.save(memoRequest.registToEntity(user, item));
        } else {
            memoRepository.save(memoRequest.registToEntity(user, item, memoRequest.getNewFile()));
        }
    }

    @Transactional
    public void updateMemo(MemoRequest memoRequest, Long memoId, Long itemSeq) throws MemoException {
        Memo memo = memoRepository.findById(memoId)
                .orElseThrow(() -> new MemoException(NO_MEMO));
        User user = userRepository.findByUserSeqAndIsDeletedFalse(memo.getUser().getUserSeq())
                .orElseThrow(() -> new NoUserException(NO_USER_EMAIL));
        // TODO : itemException 생성 후 exception 변경 예정
        Item item = itemRepository.findByItemSeq(itemSeq)
                .orElseThrow(() -> new NullPointerException(NO_FIND_ITEM));

        if(memo.getFile() == null || (memoRequest.getNewFile() != null && memo.getFile() != null )){
            memoRepository.save(memoRequest.updateToEntity(memoId,memoRequest,user,item));
        }
        else {
            memoRepository.save(memoRequest.updateToNoChangeFileEntity(memoId,memoRequest,user,item, memo.getFile()));
        }
    }

    @Transactional
    public void fakeDeleteMemo(Long memoId, MemoRequest memoRequest, Long itemSeq) throws MemoException {
        Memo memo = memoRepository.findById(memoId)
                .orElseThrow(() -> new MemoException(NO_MEMO));
        User user = userRepository.findByUserSeqAndIsDeletedFalse(memo.getUser().getUserSeq())
                .orElseThrow(() -> new NoUserException(NO_USER_EMAIL));
        // TODO : itemException 생성 후 exception 변경 예정
        Item item = itemRepository.findByItemSeq(itemSeq)
                .orElseThrow(() -> new NullPointerException(NO_FIND_ITEM));

        memoRepository.save(memoRequest.deleteToEntity(memo, user,item));
        List<Bookmark> bookmarkList = bookMarkRepository.bookmarkExistCheck(memoId,user.getUserSeq());
        if(!bookmarkList.isEmpty()) {
            deleteBookmark(memoId, user.getUserSeq());
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
        Bookmark bookmark = bookMarkRepository.findByMemoAndUser(memoId, user.getEmail())
                .orElseThrow(() -> new BookmarkException(NO_FIND_BOOKMARK));

        bookMarkRepository.delete(bookmark);
    }
}
